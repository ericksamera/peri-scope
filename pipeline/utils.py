# pipeline/utils.py

import numpy as np
from pathlib import Path
from pipeline.logger import get_logger

from PIL import Image

import torch
from torchvision.models import resnet18
import torch.nn as nn
import config

log = get_logger(__name__)


def load_tiff_channels(tif_path):
    img = Image.open(tif_path)
    img.seek(0)
    protein = np.array(img.copy())
    img.seek(1)
    membrane = np.array(img.copy())
    return protein, membrane


def normalize_image(img):
    method = config.NORM_METHOD.lower()
    img = img.astype(np.float32)

    if method == "minmax":
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    elif method == "percentile":
        pmin = np.percentile(img, config.PERCENTILE_MIN)
        pmax = np.percentile(img, config.PERCENTILE_MAX)
        return np.clip((img - pmin) / (pmax - pmin + 1e-8), 0, 1)

    elif method == "zscore":
        mean = img.mean()
        std = img.std()
        return (img - mean) / (std + 1e-8)

    elif method == "mad":
        median = np.median(img)
        mad = np.median(np.abs(img - median))
        return (img - median) / (mad + 1e-8)

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def get_next_version_id(versions_root, prefix=""):
    versions_root = Path(versions_root)
    versions_root.mkdir(parents=True, exist_ok=True)

    version_dirs = [p.name for p in versions_root.iterdir() if p.is_dir()]
    version_nums = [int(d) for d in version_dirs if d.isdigit()]
    next_id = max(version_nums, default=0) + 1
    return f"{next_id:03d}"

def load_cnn_model(model_path, device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
