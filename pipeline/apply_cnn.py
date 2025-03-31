# pipeline/apply_cnn.py

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
from torchvision import transforms
from sklearn.metrics import classification_report
import json
from datetime import datetime

from pipeline.utils import load_cnn_model

from config import get_output_paths
from pipeline.logger import get_logger
from evaluation.evaluate import evaluate_pairs

log = get_logger(__name__)


def apply_cnn_model(csv_path, model_path, device="cpu", threshold=0.5, evaluate=True):
    csv_path = Path(csv_path).resolve()
    model_path = Path(model_path).resolve()

    df = pd.read_csv(csv_path)
    crop_dir = csv_path.parent / "crops"
    df = df[df["cell_label"].notna()]
    img_paths = df["cell_label"].apply(lambda cid: crop_dir / f"cell_{int(cid):03d}.png")

    # Extract run/version IDs
    version_dir = csv_path.parent
    versions_root = version_dir.parent
    run_dir = versions_root.parent
    run_id = run_dir.name.split("_")[-1]
    version_id = get_next_version_id(versions_root)

    # Create new output paths
    paths = get_output_paths(run_id, version_id)
    paths["version"].mkdir(parents=True, exist_ok=True)
    paths["crops"].mkdir(parents=True, exist_ok=True)

    # Prepare CNN

    model = load_cnn_model(model_path, device)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # Score and annotate
    scores = []
    for idx, (cid, crop_path) in enumerate(zip(df["cell_label"], img_paths)):
        if not crop_path.exists():
            log.warning(f"Missing crop: {crop_path}")
            scores.append(np.nan)
            continue

        img = Image.open(crop_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1)[0, 1].item()
        scores.append(prob)

        # Overlay
        label = "good" if prob >= threshold else "bad"
        color = (0, 255, 0) if label == "good" else (255, 0, 0)
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), f"{label} ({prob:.2f})", fill=color + (255,))
        save_path = paths["crops"] / f"cell_{int(cid):03d}_cnn_overlay.png"
        img.save(save_path)
        log.debug(f"Overlay saved: {save_path}")

    df["score"] = scores
    out_csv = paths["csv"]
    df.to_csv(out_csv, index=False)
    log.info(f"✅ Rescored {df.shape[0]} cells with CNN and saved to {out_csv}")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "version_id": version_id,
        "model_path": str(model_path),
        "input_csv": str(csv_path),
        "output_csv": str(out_csv),
        "threshold": threshold,
        "device": device
    }

    with open(paths["version"] / "cnn_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Saved CNN scoring metadata to {paths['version'] / 'cnn_metadata.json'}")

    # Optional evaluation
    if evaluate:
        try:
            evaluate_pairs(out_csv)
        except Exception as e:
            log.warning(f"Evaluation failed: {e}")


def get_next_version_id(versions_root, prefix="version"):
    versions_root = Path(versions_root)
    versions_root.mkdir(parents=True, exist_ok=True)
    version_dirs = [p.name for p in versions_root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    version_nums = [int(d[len(prefix) + 1:]) for d in version_dirs if d[len(prefix):].lstrip("_").isdigit()]
    next_id = max(version_nums, default=0) + 1
    return f"{next_id:03d}"
