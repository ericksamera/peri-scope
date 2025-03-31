import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np
from skimage.filters import frangi
from skimage.measure import regionprops

from config import get_output_paths, MIN_CELL_AREA
from pipeline.logger import get_logger
from pipeline.export import export_protein_ring_pairs
from pipeline.utils import get_next_version_id, normalize_image
from pipeline.scorer import MLScorer
from pipeline.sample import CellSample
from pipeline.ring_detection import extract_rings  # Required for rescoring

log = get_logger(__name__)

NON_FEATURE_COLS = {
    "label", "score", "cell_label",
    "run_id", "version_id", "condition", "source_img",
    "bbox_minr", "bbox_minc", "bbox_maxr", "bbox_maxc"
}

def train_model_with_grid(csv_path, rescore=False):

    csv_path = Path(csv_path).resolve()
    version_dir = csv_path.parent
    versions_dir = version_dir.parent
    run_dir = versions_dir.parent
    run_id = run_dir.name

    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column.")

    df = df.dropna(subset=["label"])
    y = LabelEncoder().fit_transform(df["label"])
    X = df.drop(columns=[c for c in df.columns if c in NON_FEATURE_COLS], errors="ignore")

    log.info(f"Training with grid search on {len(X)} labeled samples...")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", None],
        "class_weight": [None, "balanced"],
    }

    clf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_

    log.info("Best parameters:")
    for k, v in clf.best_params_.items():
        log.info(f"  {k}: {v}")

    preds = best_model.predict(X_val)
    log.info("Validation report:")
    print(classification_report(y_val, preds))

    if rescore:
        new_version_id = get_next_version_id(run_dir / "versions")
        log.info(f"Applying trained model to same run for rescoring (→ version {new_version_id}) in {versions_dir}")
        _apply_model_to_run(best_model, list(X.columns), run_id, new_version_id)

        # 🚨 This is the fix: now we redirect saving into the rescored version folder
        version_dir = run_dir / "versions" / f"{new_version_id}"
        version_id = new_version_id
    else:
        version_id = version_dir.name

    # ✅ Save model and metadata into the appropriate version folder
    model_path = version_dir / "model.pkl"
    joblib.dump({"model": best_model, "feature_order": list(X.columns)}, model_path)
    log.info(f"Best model saved to {model_path}")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "csv_used": str(csv_path),
        "model_path": str(model_path),
        "features": list(X.columns),
        "params": best_model.get_params(),
        "grid": param_grid,
        "run_id": run_id,
        "version_id": version_id
    }

    with open(version_dir / "train_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Training metadata saved to {version_dir / 'train_metadata.json'}")


def _apply_model_to_run(model, feature_order, run_id, version_id):
    paths = get_output_paths(run_id, version_id)
    paths["version"].mkdir(parents=True, exist_ok=True)

    norm_protein = np.load(paths["run"] / "protein_img.npy")
    membrane_img = np.load(paths["run"] / "membrane_img.npy")
    cell_labels = np.load(paths["run"] / "cell_labels.npy")
    frangi_img = frangi(normalize_image(membrane_img))

    samples = [
        CellSample(region.label, region, cell_labels == region.label)
        for region in regionprops(cell_labels)
        if region.area >= MIN_CELL_AREA
    ]

    scorer = MLScorer(model, feature_order)

    # ✅ Ensure rescored samples get fresh rings
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
        version_id=version_id
    )
