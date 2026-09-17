"""40-phon table and historical nearest-neighbor lookup."""
import torch

SPL_40_PHON = {20: 99.85, 25: 93.94, 31.5: 88.17, 40: 82.63, 50: 77.78, 63: 73.08, 80: 68.48, 100: 64.37, 125: 60.59, 160: 56.7, 200: 53.41, 250: 50.4, 315: 47.58, 400: 44.98, 500: 43.05, 630: 41.34, 800: 40.06, 1000: 40.01, 1250: 41.82, 1600: 42.51, 2000: 39.23, 2500: 36.51, 3150: 35.61, 4000: 36.65, 5000: 40.01, 6300: 45.83, 8000: 51.8, 10000: 54.28, 12500: 51.49}

def freq_to_spl(freq, phon_level=40):
    if phon_level != 40:
        raise ValueError("Only the paper's 40-phon contour is supported")
    def lookup(f):
        return SPL_40_PHON[min(SPL_40_PHON, key=lambda x: abs(x - f))]
    if isinstance(freq, torch.Tensor):
        return torch.tensor([lookup(f.item()) for f in freq.flatten()], dtype=freq.dtype, device=freq.device).reshape(freq.shape)
    return lookup(freq)
