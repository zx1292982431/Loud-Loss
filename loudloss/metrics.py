"""Reference metrics with explicit names (historical SDR is not renamed SNR)."""
import torch
from torchmetrics.functional.audio import (
    signal_distortion_ratio, scale_invariant_signal_distortion_ratio,
    signal_noise_ratio, scale_invariant_signal_noise_ratio,
    perceptual_evaluation_speech_quality, short_time_objective_intelligibility,
)


def compute_metrics(pred, target):
    pred, target = pred.cpu().reshape(1, -1), target.cpu().reshape(1, -1)
    values = {
        "wb_pesq": perceptual_evaluation_speech_quality(pred, target, 16000, "wb"),
        "nb_pesq": perceptual_evaluation_speech_quality(pred, target, 16000, "nb"),
        "stoi": short_time_objective_intelligibility(pred, target, 16000),
        "estoi": short_time_objective_intelligibility(pred, target, 16000, extended=True),
        "sdr": signal_distortion_ratio(pred, target),
        "si_sdr": scale_invariant_signal_distortion_ratio(pred, target),
        "snr": signal_noise_ratio(pred, target),
        "si_snr": scale_invariant_signal_noise_ratio(pred, target),
    }
    result = {k: float(v.mean()) for k, v in values.items()}
    if not all(torch.isfinite(torch.tensor(v)) for v in result.values()):
        raise ValueError(f"Non-finite evaluation metric: {result}")
    return result
