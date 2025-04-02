# pipeline/train_helpers.py
import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path
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

from PIL import Image
from pipeline.visualize import tile_images

from pipeline.segmentation import segment_cells
from pipeline.utils import load_tiff_channels

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

    # 🔍 Find detect-export metadata
    versions_dir = paths["run"] / "versions"
    source_meta = None
    for version_path in sorted(versions_dir.glob("*/source_metadata.json")):
        with open(version_path) as f:
            meta = json.load(f)
            if "inputs" in meta:
                source_meta = meta
                break

    if not source_meta:
        raise FileNotFoundError(f"Could not find source_metadata.json in {versions_dir}")

    tiff_paths = [Path(entry["tif"]) for entry in source_meta.get("inputs", [])]

    scorer = MLScorer(model, feature_order)
    all_samples = []

    for tif_path in tiff_paths:
        log.info(f"🔁 Rescoring {tif_path}")
        protein_img, membrane_img = load_tiff_channels(tif_path)
        norm_membrane = normalize_image(membrane_img)
        norm_protein = normalize_image(protein_img)
        frangi_img = frangi(norm_membrane)

        cell_labels = segment_cells(norm_membrane, debug=False)
        samples = []
        for region in regionprops(cell_labels):
            if region.area >= MIN_CELL_AREA:
                sample = CellSample(cell_id=region.label, region=region, cell_mask=(cell_labels == region.label))
                sample._membrane_img = membrane_img
                sample._norm_protein = norm_protein
                sample._frangi_img = frangi_img

                sample.meta.update({
                    "source_img": str(tif_path),
                    "run_id": run_id,
                    "version_id": version_id,
                })

                samples.append(sample)

        ring_mask = extract_rings(
            membrane_img, samples, norm_protein, scorer, frangi_img, debug=False
        )

        for sample in samples:
            sample.meta.update({
                "source_img": str(tif_path),
                "run_id": run_id,
                "version_id": version_id,
            })

        all_samples.extend(samples)

    export_protein_ring_pairs(
        samples=all_samples,
        scorer=scorer,
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
            merged = merged.drop_duplicates(subset=["cell_label"], keep="first")
            merged.to_csv(paths["csv"], index=False)
            if "label" in merged.columns:
                cols = merged.columns.tolist()
                cols.insert(0, cols.pop(cols.index("label")))
                merged = merged[cols]
                merged.to_csv(paths["csv"], index=False)
            log.info(f"Restored metadata columns from {original_csv}")
        else:
            log.warning(f"No metadata columns found to restore from {original_csv}")

    out_dir = Path("outputs") / run_id / "versions" / version_id
    csv_path = out_dir / "pairs_metadata.csv"
    crop_dir = Path("outputs") / run_id / "crops"

    df = pd.read_csv(csv_path)
    top = df.sort_values("score", ascending=False).head(100)
    thumbs = []
    for _, row in top.iterrows():
        cell_id = int(row["cell_label"])
        crop_path = crop_dir / f"cell_{cell_id:04d}.png"
        if crop_path.exists():
            thumbs.append(Image.open(crop_path))

    if thumbs:
        tile = tile_images(thumbs, rows=10, cols=10)
        tile.save(out_dir / "top100_tile.png")
        print(f"🖼️  Saved top 100 tile to: {out_dir / 'top100_tile.png'}")