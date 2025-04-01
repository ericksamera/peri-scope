# pipeline/train_helpers.py
import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path
import numpy as np

from sklearn.preprocessing import LabelEncoder

from config import get_output_paths, MIN_CELL_AREA
from pipeline.logger import get_logger
from pipeline.export import export_protein_ring_pairs
from pipeline.utils import normalize_image
from pipeline.scorer import MLScorer
from pipeline.sample import CellSample
from pipeline.ring_detection import extract_rings
from skimage.filters import frangi
from skimage.measure import regionprops

log = get_logger(__name__)

NON_FEATURE_COLS = {
    "label", "score", "cell_label",
    "run_id", "version_id", "condition", "source_img",
    "bbox_minr", "bbox_minc", "bbox_maxr", "bbox_maxc"
}

def prepare_features(csv_path):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column.")

    df = df.dropna(subset=["label"])
    y = LabelEncoder().fit_transform(df["label"])
    X = df.drop(columns=[c for c in df.columns if c in NON_FEATURE_COLS], errors="ignore")
    X = X.select_dtypes(include=[float, int])
    return X, y, df

def save_model_and_metadata(model, X_cols, csv_path, run_id, version_id, model_path_suffix="model.pkl", extra_metadata=None):
    version_dir = Path(get_output_paths(run_id, version_id)["version"])
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / model_path_suffix
    joblib.dump({"model": model, "feature_order": list(X_cols)}, model_path)
    log.info(f"Model saved to {model_path}")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "csv_used": str(csv_path),
        "model_path": str(model_path),
        "features": list(X_cols),
        "run_id": run_id,
        "version_id": version_id,
        **(extra_metadata or {})
    }

    with open(version_dir / "train_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata saved to {version_dir / 'train_metadata.json'}")

def apply_model_to_run(model, feature_order, run_id, version_id, original_csv=None, skip_crops=True):
    paths = get_output_paths(run_id, version_id)
    paths["version"].mkdir(parents=True, exist_ok=True)

    norm_protein = np.load(paths["run"] / "protein_img.npy")
    membrane_img = np.load(paths["run"] / "membrane_img.npy")
    cell_labels = np.load(paths["run"] / "cell_labels.npy")
    frangi_img = frangi(normalize_image(membrane_img))

    samples = [
        CellSample.from_region(region, cell_labels)
        for region in regionprops(cell_labels)
        if region.area >= MIN_CELL_AREA
    ]

    scorer = MLScorer(model, feature_order)

    ring_mask = extract_rings(
        membrane_img, samples, norm_protein, scorer, frangi_img, debug=False
    )

    export_protein_ring_pairs(
        samples,
        membrane_img,
        norm_protein,
        frangi_img,
        scorer,
        paths=paths,
        condition_label=None,
        source_img="rescored_from_model",
        run_id=run_id,
        version_id=version_id,
        skip_crops=skip_crops,
    )

    if original_csv and Path(original_csv).exists():
        df_old = pd.read_csv(original_csv)
        df_new = pd.read_csv(paths["csv"])

        merge_cols = [c for c in ["label", "condition", "source_img"] if c in df_old.columns]
        if merge_cols:
            merged = pd.merge(df_new, df_old[["cell_label"] + merge_cols], on="cell_label", how="left")
            merged.to_csv(paths["csv"], index=False)
            log.info(f"Restored metadata columns from {original_csv}")
        else:
            log.warning(f"No metadata columns found to restore from {original_csv}")
