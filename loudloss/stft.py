"""The paper checkpoints' centered, reflect-padded Hann STFT."""
import torch
from torch import nn


class STFT(nn.Module):
    def __init__(self, n_fft=512, n_hop=256, win_len=512):
        super().__init__()
        self.n_fft, self.n_hop, self.win_len = n_fft, n_hop, win_len
        self.register_buffer("window", torch.hann_window(win_len))

    def stft(self, x):
        shape = x.shape
        spec = torch.stft(x.reshape(-1, shape[-1]), self.n_fft, self.n_hop,
                          self.win_len, self.window, return_complex=True)
        return spec.reshape(*shape[:-1], *spec.shape[-2:]), shape[-1]

    def istft(self, spec, length):
        shape = spec.shape
        # Keep the historical per-sample inverse transform.
        waves = [torch.istft(s, self.n_fft, self.n_hop, self.win_len,
                             self.window, length=length)
                 for s in spec.reshape(-1, *shape[-2:])]
        return torch.stack(waves).reshape(*shape[:-2], length)

    def forward(self, x):
        return self.stft(x)

    def stftBFTH(self, x):
        """Historical real-valued [B,F,T,2C] view used by the MSE objective."""
        spec, length = self.stft(x)
        spec = spec.permute(0, 2, 3, 1)
        b, f, t, _ = spec.shape
        return torch.view_as_real(spec).reshape(b, f, t, -1), length
