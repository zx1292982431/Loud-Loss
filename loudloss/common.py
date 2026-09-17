import importlib.metadata
import platform
import os
from pathlib import Path
import torch
import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    if config["sample_rate"] != 16000:
        raise ValueError("Paper models only support 16 kHz")
    return config


def checkpoint_path(args, config):
    return Path(args.checkpoint) if args.checkpoint else ROOT / config["checkpoint"]


def environment(device):
    packages = ["torch", "torchmetrics", "pesq", "pystoi", "numpy", "scipy", "soundfile"]
    return {"python": platform.python_version(), "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device) if str(device).startswith("cuda") else None,
            "threads": torch.get_num_threads(),
            "packages": {p: importlib.metadata.version(p) for p in packages}}
