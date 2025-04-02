import pandas as pd
from pipeline.feature_extraction import extract_ring_features
from pipeline.logger import get_logger
from pipeline.visualize import save_ring_crop
from typing import List
from pipeline.sample import CellSample

from concurrent.futures import ThreadPoolExecutor

log = get_logger(__name__)

def export_protein_ring_pairs(
    samples: List[CellSample],
    scorer,
    paths,
    condition_label=None,
    source_img=None,
    run_id=None,
    version_id=None,
    skip_crops=False,
):
    crop_dir = paths["crops"]
    csv_path = paths["csv"]
    crop_dir.mkdir(parents=True, exist_ok=True)

    ring_samples = [s for s in samples if s.has_ring()]
    if not ring_samples:
        log.warning("No ring-positive cells found. Skipping export.")
        return

    rows = []
    for sample in ring_samples:
        features = extract_ring_features(
            sample.region,
            sample._membrane_img,
            sample._norm_protein,
            sample.ring_mask,
            sample._frangi_img,
        )
        if features is None:
            log.debug(f"Cell {sample.cell_id}: No features extracted.")
            continue

        sample.set_features(features)
        score = scorer.score(features)
        sample.set_score(score)

        sample.meta.update({
            "condition": condition_label,
            "source_img": source_img,
            "run_id": run_id,
            "version_id": version_id,
        })

        rows.append(sample.to_row())

    if not rows:
        log.warning("No protein+ring pairs were exported.")
        return

    df = pd.DataFrame(rows)
    if "label" in df.columns:
        cols = df.columns.tolist()
        cols.insert(0, cols.pop(cols.index("label")))
        df = df[cols]

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True).drop_duplicates(subset=["cell_label"])
    else:
        df_combined = df

    df_combined.to_csv(csv_path, index=False)
    log.info(f"✅ Exported {len(df)} new cells to {csv_path.resolve()}")

    if not skip_crops:
        max_id = max(s.cell_id for s in ring_samples)
        pad_width = max(4, len(str(max_id)))

        def export_crop(sample):
            out_path = crop_dir / f"cell_{sample.cell_id:0{pad_width}d}.png"
            save_ring_crop(sample._norm_protein, sample.ring_mask, sample.cell_mask, out_path)

        with ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(export_crop, ring_samples)