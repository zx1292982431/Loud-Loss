"""Arrange already-resampled 16 kHz VoiceBank audio according to release manifests."""
import argparse
import csv
from pathlib import Path
import shutil
from loudloss.common import ROOT
from loudloss.data import read_audio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=["test", "train", "all"], default="test")
    args = p.parse_args()
    source, output = Path(args.source_root), Path(args.output_dir)
    splits = ["test", "train", "unused_validation"] if args.split == "all" else [args.split]
    planned = []
    for split in splits:
        with open(ROOT / f"manifests/{split}.tsv") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                for kind in ("noisy", "clean"):
                    rel = Path(row[kind])
                    official = f"{kind}_testset_wav" if split == "test" else f"{kind}_trainset_28spk_wav"
                    candidates = [source / rel, source / official / rel.name]
                    existing = [x for x in candidates if x.is_file()]
                    if not existing:
                        raise FileNotFoundError(f"Missing {rel.name}; tried {candidates}")
                    dest = output / rel
                    if dest.exists():
                        raise FileExistsError(f"Refusing to overwrite {dest}")
                    read_audio(existing[0])  # Reject 48 kHz, stereo and malformed data before copying.
                    planned.append((existing[0], dest))
    for src, dest in planned:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    print(f"Copied {len(planned)} files; no resampling or amplitude changes")


if __name__ == "__main__":
    main()
