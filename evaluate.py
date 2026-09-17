"""Evaluate complete utterances; never skip failures or silently change audio."""
import argparse
import json
from pathlib import Path
import time
import torch
from loudloss.model import Enhancer
from loudloss.data import VoiceBank
from loudloss.metrics import compute_metrics
from loudloss.common import ROOT, load_config, checkpoint_path, environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--data-root", required=True, help="Directory containing test_noisy/ and test_clean/")
    p.add_argument("--manifest", default=str(ROOT / "manifests/test.tsv"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    ckpt = checkpoint_path(args, config)
    model = Enhancer().load_checkpoint(ckpt).to(args.device).eval()
    dataset = VoiceBank(args.data_root, args.manifest)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results_mean.json").exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    count = min(args.limit or len(dataset), len(dataset))
    rows, start = [], time.monotonic()
    with torch.inference_mode(), open(output / "per_utterance.jsonl", "w") as f:
        for i in range(count):
            noisy, clean, name = dataset[i]
            pred = model(noisy[None, None].to(args.device))[0, 0].cpu()
            row = {"id": i, "filename": name, "samples": len(noisy),
                   **compute_metrics(pred, clean),
                   **{"input_" + k: v for k, v in compute_metrics(noisy, clean).items()}}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if (i + 1) % 50 == 0 or i + 1 == count:
                print(f"{config['experiment']}: {i+1}/{count}, {time.monotonic()-start:.1f}s", flush=True)
    metric_keys = [k for k in rows[0] if k not in {"id", "filename", "samples"}]
    summary = {k: sum(r[k] for r in rows) / count for k in metric_keys}
    (output / "results_mean.json").write_text(json.dumps(summary, indent=2) + "\n")
    metadata = {"experiment": config["experiment"], "count": count,
                "complete": count == len(dataset),
                "elapsed_seconds": time.monotonic()-start, **environment(args.device)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
