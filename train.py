"""Train the three paper objectives with the historical padded-batch protocol."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.audio import perceptual_evaluation_speech_quality
import yaml
from loudloss import Enhancer, LoudLoss, LoudLossSISNR, MagMSELoss
from loudloss.data import VoiceBank, collate
from loudloss.common import ROOT, load_config, environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", help="Initialize model weights only; does not restore optimizer")
    p.add_argument("--resume", help="Resume a checkpoint produced by this train.py")
    p.add_argument("--data-root", required=True)
    p.add_argument("--train-manifest", default=str(ROOT / "manifests/train.tsv"))
    p.add_argument("--val-manifest", default=str(ROOT / "manifests/test.tsv"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--limit-train-batches", type=int)
    p.add_argument("--limit-val-batches", type=int)
    args = p.parse_args()
    if args.checkpoint and args.resume:
        p.error("Choose --checkpoint or --resume")
    for value in [args.epochs, args.limit_train_batches, args.limit_val_batches]:
        if value is not None and value < 1:
            p.error("Epoch and batch limits must be positive")
    config = load_config(args.config)
    if args.epochs:
        config["epochs"] = args.epochs
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)
    model = Enhancer()
    if args.checkpoint:
        model.load_checkpoint(args.checkpoint)
    model.to(args.device)
    loss_fn = {"LoudLoss": LoudLoss, "LoudLossSISNR": LoudLossSISNR,
               "MagMSELoss": MagMSELoss}[config["loss"]]().to(args.device)
    train_set = VoiceBank(args.data_root, args.train_manifest)
    val_set = VoiceBank(args.data_root, args.val_manifest)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                              num_workers=args.workers, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False,
                            num_workers=args.workers, collate_fn=collate)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config["scheduler_factor"],
        patience=config["scheduler_patience"], min_lr=config["min_lr"])
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "last.ckpt").exists():
        raise FileExistsError("Use a new output directory or --resume")
    epoch_start, step, best = 0, 0, float("-inf")
    if args.resume:
        raw = torch.load(args.resume, map_location="cpu", weights_only=True)
        same_settings = {k: v for k, v in raw.get("config", {}).items() if k != "epochs"} == {
            k: v for k, v in config.items() if k != "epochs"}
        if raw.get("release_format") != 1 or not same_settings:
            raise ValueError("Resume requires a release checkpoint with matching configuration")
        model.load_state_dict(raw["state_dict"], strict=True)
        optimizer.load_state_dict(raw["optimizer"])
        scheduler.load_state_dict(raw["scheduler"])
        epoch_start, step, best = raw["epoch"] + 1, raw["global_step"], raw["best_wb_pesq"]
        if config["epochs"] <= epoch_start:
            raise ValueError("No epochs remain; increase --epochs to extend this run")
        torch.set_rng_state(raw["torch_rng"])
        if torch.cuda.is_available() and raw["cuda_rng"]:
            torch.cuda.set_rng_state_all(raw["cuda_rng"])
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {**environment(args.device), "training_samples": len(train_set),
                "validation_samples": len(val_set), "validation_uses_test_manifest":
                Path(args.val_manifest).read_bytes() == (ROOT / "manifests/test.tsv").read_bytes(),
                "limit_train_batches": args.limit_train_batches,
                "limit_val_batches": args.limit_val_batches}
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with SummaryWriter(output / "tensorboard", purge_step=step if args.resume else None) as writer:
        for epoch in range(epoch_start, config["epochs"]):
            model.train()
            for batch_idx, (noisy, clean, _) in enumerate(train_loader):
                if args.limit_train_batches and batch_idx >= args.limit_train_batches:
                    break
                noisy, clean = noisy.to(args.device), clean.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = loss_fn(model(noisy), clean)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                writer.add_scalar("train/" + config["loss"], loss.item(), step)
                step += 1
            model.eval()
            val_loss, val_pesq, samples = 0.0, 0.0, 0
            with torch.no_grad():
                for batch_idx, (noisy, clean, _) in enumerate(val_loader):
                    if args.limit_val_batches and batch_idx >= args.limit_val_batches:
                        break
                    noisy, clean = noisy.to(args.device), clean.to(args.device)
                    pred = model(noisy)
                    loss, _ = loss_fn(pred, clean)
                    # Preserve validation PESQ on the full zero-padded batch.
                    pesq = perceptual_evaluation_speech_quality(pred.cpu(), clean.cpu(), 16000, "wb").mean()
                    if not torch.isfinite(loss) or not torch.isfinite(pesq):
                        raise ValueError("Non-finite validation result")
                    val_loss += loss.item() * len(noisy)
                    val_pesq += pesq.item() * len(noisy)
                    samples += len(noisy)
            val_loss, val_pesq = val_loss / samples, val_pesq / samples
            scheduler.step(val_loss)
            improved = val_pesq > best
            best = max(best, val_pesq)
            for tag, value in [("val/loss", val_loss), ("val/wb_pesq", val_pesq),
                               ("lr", optimizer.param_groups[0]["lr"])]:
                writer.add_scalar(tag, value, epoch)
            state = {"release_format": 1, "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "epoch": epoch, "global_step": step, "best_wb_pesq": best,
                     "config": config, "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
            torch.save(state, output / "last.ckpt")
            if improved:
                torch.save(state, output / "best.ckpt")
            print(f"epoch={epoch} val_loss={val_loss:.6f} val_wb_pesq={val_pesq:.6f}", flush=True)


if __name__ == "__main__":
    main()
