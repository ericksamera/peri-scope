import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np
from skimage.measure import regionprops
from skimage.filters import frangi

from config import get_output_paths, MIN_CELL_AREA
from pipeline.logger import get_logger
from pipeline.export import export_protein_ring_pairs
from pipeline.utils import get_next_version_id, normalize_image
from pipeline.scorer import MLScorer
from pipeline.sample import CellSample
from pipeline.ring_detection import extract_rings  # required for rescoring

log = get_logger(__name__)


def train_classifier(csv_path, rescore=False):
    csv_path = Path(csv_path).resolve()
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must include a 'label' column.")

    df = df.dropna(subset=["label"])
    y = LabelEncoder().fit_transform(df["label"])
    X = df.select_dtypes(include=[np.number]).copy()

    log.info(f"Training on {len(X)} labeled samples...")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    log.info("Validation report:")
    print(classification_report(y_val, preds))

    # Locate run/version from current CSV path
    version_dir = csv_path.parent
    run_dir = version_dir.parent.parent
    run_id = run_dir.name

    if rescore:
        new_version_id = get_next_version_id(run_dir / "versions")
        log.info(f"Applying trained model to same run for rescoring (→ version {new_version_id})...")
        _apply_model_to_run(model, list(X.columns), run_id, new_version_id, original_csv=csv_path)

        # 🔁 Save model + metadata into rescored version
        version_dir = run_dir / "versions" / f"version_{new_version_id}"
        version_id = new_version_id
    else:
        version_id = version_dir.name

    # ✅ Save model and metadata into the active version folder
    model_path = version_dir / "model.pkl"
    joblib.dump({"model": model, "feature_order": list(X.columns)}, model_path)
    log.info(f"Trained model saved to {model_path}")

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "csv_used": str(csv_path),
        "model_path": str(model_path),
        "features": list(X.columns),
        "params": model.get_params(),
        "run_id": run_id,
        "version_id": version_id
    }

    with open(version_dir / "train_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Training metadata saved to {version_dir / 'train_metadata.json'}")


def _apply_model_to_run(model, feature_order, run_id, version_id, original_csv=None):
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

    # Extract fresh rings and score
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

    if original_csv and Path(original_csv).exists():
        df_old = pd.read_csv(original_csv)
        df_new = pd.read_csv(paths["csv"])

        cols_to_restore = ["label", "condition", "source_img"]
        merge_cols = [col for col in cols_to_restore if col in df_old.columns]

        if merge_cols:
            merged = pd.merge(df_new, df_old[["cell_label"] + merge_cols], on="cell_label", how="left")
            merged.to_csv(paths["csv"], index=False)
            log.info(f"Restored columns {merge_cols} from {original_csv} into {paths['csv']}")
        else:
            log.warning(f"No metadata columns found to restore from {original_csv}")
    else:
        log.warning(f"No original CSV found to restore metadata: {original_csv}")
