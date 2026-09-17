"""Enhance one mono 16 kHz WAV using a published checkpoint."""
import argparse
from pathlib import Path
import soundfile as sf
import torch
from loudloss.model import Enhancer
from loudloss.data import read_audio
from loudloss.common import load_config, checkpoint_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--input", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    model = Enhancer().load_checkpoint(checkpoint_path(args, load_config(args.config))).to(args.device).eval()
    audio = read_audio(args.input)
    with torch.inference_mode():
        enhanced = model(audio[None, None].to(args.device))[0, 0].cpu().numpy()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / Path(args.input).name
    if path.exists():
        raise FileExistsError(path)
    sf.write(path, enhanced, 16000, subtype="FLOAT")
    print(path)


if __name__ == "__main__":
    main()
