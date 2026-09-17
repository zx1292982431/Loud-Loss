"""Historical waveform objectives; returns (scalar loss, diagnostics)."""
import torch
from torch import nn, Tensor
import numpy
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from .stft import STFT
from .perceptual import freq_to_spl

def neg_si_sdr(preds: Tensor, target: Tensor) -> Tensor:
    """calculate neg_si_sdr loss for a batch

    Returns:
        loss: shape [batch], real
    """
    batch_size = target.shape[0]
    si_sdr_val = si_sdr(preds=preds, target=target)
    return -torch.mean(si_sdr_val.view(batch_size, -1), dim=1)


class MagMSELoss(nn.Module):
    """Legacy MSE of absolute real/imaginary components, not complex magnitudes."""
    def __init__(self, n_fft=512, hop_len=256, win_len=512):
        super().__init__()
        self.stft = STFT(n_fft, hop_len, win_len)
        self.name = 'MagMSELoss'

    def forward(self, pred, target):
        pred_spec, _ = self.stft.stftBFTH(pred)
        target_spec, _ = self.stft.stftBFTH(target)

        pred_spec = torch.abs(pred_spec)
        target_spec = torch.abs(target_spec)

        loss = nn.functional.mse_loss(pred_spec, target_spec).mean()

        loss_dict = {
            'total': loss,
            'lf': 0,
            'mf': 0,
            'hf': 0
        }

        return loss, loss_dict


class LoudLoss(nn.Module):
    def __init__(self, n_fft=512, hop_len=256, win_len=512, mel_start_hz=0, mel_end_hz=8000, mel_bins=26):
        super().__init__()
        self.stft = STFT(n_fft, hop_len, win_len)
        self.name = 'LoudLoss'
        self.mel_start = mel_start_hz
        self.mel_end = mel_end_hz
        self.mel_bins = mel_bins
        self.center_freqs = self.compute_center_freqs()
        self.spl = freq_to_spl(self.center_freqs,40)
        spl_1k = freq_to_spl(1000,40)
        self.band_weight = 1/(self.spl/spl_1k)

        self.center_freqs_stft = torch.floor(((n_fft // 2) + 1 + 1) * self.center_freqs / 8000).to(torch.int32)
        if mel_start_hz != 0:
            self.center_freqs_stft = torch.cat([torch.zeros(1), self.center_freqs_stft], dim=0).to(torch.int32)

    def compute_center_freqs(self):
        mel_start = self.hz2mel(self.mel_start)
        mel_end = self.hz2mel(self.mel_end)
        mel_bins = self.mel_bins
        mel_freqs = torch.linspace(mel_start, mel_end, mel_bins + 1)
        hz_freqs = self.mel2hz(mel_freqs)
        return hz_freqs

    def hz2mel(self, hz):
        return 2595 * numpy.log10(1 + hz / 700.0)

    def mel2hz(self, mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    def forward(self, pred, target):
        pred_spec, _ = self.stft.stft(pred)
        target_spec, _ = self.stft.stft(target)

        pow_pred_spec = torch.log(torch.pow(torch.abs(pred_spec), 2) + 1e-10) # B,1,F,T
        pow_target_spec = torch.log(torch.pow(torch.abs(target_spec), 2) + 1e-10) # B,1,F,T

        band_spilt_pow_pred_spec = []
        band_spilt_pow_target_spec = []

        total_loss = 0
        for i in range(self.mel_bins-1):
            band_spilt_pow_pred_spec.append(pow_pred_spec[:, :, self.center_freqs_stft[i]:self.center_freqs_stft[i+2]+1,:])
            band_spilt_pow_target_spec.append(pow_target_spec[:, :, self.center_freqs_stft[i]:self.center_freqs_stft[i+2]+1,:])

        loss_dict = {}
        for i in range(self.mel_bins-1):
            cur_loss = nn.functional.mse_loss(band_spilt_pow_pred_spec[i], band_spilt_pow_target_spec[i])
            total_loss += cur_loss * self.band_weight[i+1]
            loss_dict[f'band_{i}'] = cur_loss.item()

        return total_loss, loss_dict


class LoudLossSISNR(nn.Module):
    def __init__(self, n_fft=512, hop_len=256, win_len=512, mel_start_hz=0, mel_end_hz=8000, mel_bins=26):
        super().__init__()
        self.stft = STFT(n_fft, hop_len, win_len)
        self.name = 'LoudLossSISNR'
        self.mel_start = mel_start_hz
        self.mel_end = mel_end_hz
        self.mel_bins = mel_bins
        self.center_freqs = self.compute_center_freqs()
        self.spl = freq_to_spl(self.center_freqs,40)
        spl_1k = freq_to_spl(1000,40)
        self.band_weight = 1/(self.spl/spl_1k)

        self.center_freqs_stft = torch.floor(((n_fft // 2) + 1 + 1) * self.center_freqs / 8000).to(torch.int32)
        if mel_start_hz != 0:
            self.center_freqs_stft = torch.cat([torch.zeros(1), self.center_freqs_stft], dim=0).to(torch.int32)

    def compute_center_freqs(self):
        mel_start = self.hz2mel(self.mel_start)
        mel_end = self.hz2mel(self.mel_end)
        mel_bins = self.mel_bins
        mel_freqs = torch.linspace(mel_start, mel_end, mel_bins + 1)
        hz_freqs = self.mel2hz(mel_freqs)
        return hz_freqs

    def hz2mel(self, hz):
        return 2595 * numpy.log10(1 + hz / 700.0)

    def mel2hz(self, mel):
        return 700 * (10 ** (mel / 2595.0) - 1)

    def forward(self, pred, target):
        pred_spec, _ = self.stft.stft(pred)
        target_spec, _ = self.stft.stft(target)

        sisnr_loss = neg_si_sdr(pred, target).mean()

        pow_pred_spec = torch.log(torch.pow(torch.abs(pred_spec), 2) + 1e-10) # B,1,F,T
        pow_target_spec = torch.log(torch.pow(torch.abs(target_spec), 2) + 1e-10) # B,1,F,T

        band_spilt_pow_pred_spec = []
        band_spilt_pow_target_spec = []

        total_loss = 0
        for i in range(self.mel_bins-1):
            band_spilt_pow_pred_spec.append(pow_pred_spec[:, :, self.center_freqs_stft[i]:self.center_freqs_stft[i+2]+1,:])
            band_spilt_pow_target_spec.append(pow_target_spec[:, :, self.center_freqs_stft[i]:self.center_freqs_stft[i+2]+1,:])

        loss_dict = {}
        for i in range(self.mel_bins-1):
            cur_loss = nn.functional.mse_loss(band_spilt_pow_pred_spec[i], band_spilt_pow_target_spec[i])
            total_loss += cur_loss * self.band_weight[i+1]
            loss_dict[f'band_{i}'] = cur_loss.item()

        total_loss = total_loss * 0.1 + sisnr_loss
        return total_loss, loss_dict
