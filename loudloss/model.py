"""Waveform enhancement with the original magnitude mapping and noisy phase."""
import torch
from torch import nn
from .gtcrn import GTCRNMapping
from .stft import STFT


class Enhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.arch = GTCRNMapping()
        self.stft = STFT()

    def forward(self, wave):
        """Accept [batch, 1, samples], return the same shape; 16 kHz only."""
        if wave.ndim != 3 or wave.shape[1] != 1 or wave.shape[-1] <= 256:
            raise ValueError("Expected mono [batch, 1, samples] with >256 samples")
        spec, length = self.stft.stft(wave)
        features = torch.view_as_real(spec[:, 0])
        pred = self.arch(features)
        pred = torch.view_as_complex(pred.float().contiguous()).unsqueeze(1)
        return self.stft.istft(pred, length)

    def load_checkpoint(self, path):
        """Read published tensor weights or original Lightning checkpoints."""
        raw = torch.load(path, map_location="cpu", weights_only=True)
        state = raw.get("state_dict", raw)
        if any(k.startswith("arch.") for k in state):
            extra = set(state) - {k for k in state if k.startswith("arch.")}
            if extra - {"stft.window", "loss.stft.window"}:
                raise ValueError(f"Unexpected checkpoint keys: {sorted(extra)}")
            for key in extra:
                if not torch.equal(state[key], self.stft.window):
                    raise ValueError(f"Incompatible STFT window: {key}")
            state = {k[5:]:v for k, v in state.items() if k.startswith("arch.")}
        self.arch.load_state_dict(state, strict=True)
        return self
