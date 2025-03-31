# pipeline/utils.py

import numpy as np
from pathlib import Path
from pipeline.logger import get_logger

import torch
from torchvision.models import resnet18
import torch.nn as nn

log = get_logger(__name__)

def normalize_image(img: np.ndarray) -> np.ndarray:
    log.debug(f"Normalizing image of shape {img.shape}")
    return (img - img.min()) / (np.ptp(img) + 1e-6)

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
