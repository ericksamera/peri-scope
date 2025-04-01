# pipeline/train_grid.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

from pipeline.train_helpers import (
    prepare_features,
    save_model_and_metadata,
    apply_model_to_run,
)
from pipeline.utils import get_next_version_id
from pipeline.logger import get_logger
from pathlib import Path

log = get_logger(__name__)

def train_model_with_grid(csv_path, rescore=False):
    csv_path = Path(csv_path).resolve()
    X, y, df = prepare_features(csv_path)

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

    version_dir = csv_path.parent
    run_dir = version_dir.parent.parent
    run_id = run_dir.name

    if rescore:
        version_id = get_next_version_id(run_dir / "versions")
        log.info(f"Applying best model to same run (→ version {version_id})...")
        apply_model_to_run(best_model, list(X.columns), run_id, version_id, original_csv=csv_path, skip_crops=True)
    else:
        version_id = version_dir.name

    save_model_and_metadata(
        best_model,
        X.columns,
        csv_path,
        run_id,
        version_id,
        extra_metadata={"grid": param_grid}
    )
