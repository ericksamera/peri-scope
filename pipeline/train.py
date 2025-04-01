# pipeline/train.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from pipeline.train_helpers import (
    prepare_features,
    save_model_and_metadata,
    apply_model_to_run,
)
from pipeline.utils import get_next_version_id
from pipeline.logger import get_logger
from pathlib import Path

log = get_logger(__name__)

def train_classifier(csv_path, rescore=False):
    csv_path = Path(csv_path).resolve()
    X, y, df = prepare_features(csv_path)

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

    # Determine where to save
    version_dir = csv_path.parent
    run_dir = version_dir.parent.parent
    run_id = run_dir.name

    if rescore:
        version_id = get_next_version_id(run_dir / "versions")
        log.info(f"Applying trained model to same run (→ version {version_id})...")
        apply_model_to_run(model, list(X.columns), run_id, version_id, original_csv=csv_path, skip_crops=True)
    else:
        version_id = version_dir.name

    save_model_and_metadata(model, X.columns, csv_path, run_id, version_id)
