"""Portable paired VoiceBank manifests. No automatic resampling or normalization."""
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000 or audio.ndim != 1:
        raise ValueError(f"Expected mono 16 kHz audio: {path} ({sr} Hz, {audio.shape})")
    if len(audio) <= 256 or not np.isfinite(audio).all():
        raise ValueError(f"Audio is too short or non-finite: {path}")
    return torch.from_numpy(audio)


class VoiceBank(Dataset):
    def __init__(self, data_root, manifest):
        self.root = Path(data_root)
        with open(manifest) as f:
            self.rows = list(csv.DictReader(f, delimiter="\t"))
        if not self.rows:
            raise ValueError("Empty manifest")
        seen = set()
        for row in self.rows:
            a, b = Path(row["noisy"]), Path(row["clean"])
            if a.is_absolute() or b.is_absolute() or ".." in a.parts or ".." in b.parts:
                raise ValueError("Manifest paths must be relative to data-root")
            if a.name != b.name or a.name in seen:
                raise ValueError(f"Unpaired or duplicate utterance: {row}")
            seen.add(a.name)
            for path in (a, b):
                if not (self.root / path).is_file():
                    raise FileNotFoundError(self.root / path)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        noisy, clean = (read_audio(self.root / row[k]) for k in ("noisy", "clean"))
        if noisy.shape != clean.shape:
            raise ValueError(f"Length mismatch: {row}")
        return noisy, clean, Path(row["noisy"]).name


def collate(batch):
    # Historical training/validation losses include zero padding.
    batch = sorted(batch, key=lambda row: len(row[0]), reverse=True)
    noisy, clean, names = zip(*batch)
    return (pad_sequence(noisy, batch_first=True).unsqueeze(1),
            pad_sequence(clean, batch_first=True).unsqueeze(1), names)
