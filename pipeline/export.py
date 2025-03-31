# pipeline/export.py

import pandas as pd
from pipeline.feature_extraction import extract_ring_features
from pipeline.logger import get_logger
from pipeline.visualize import save_ring_crop

from typing import List
from pipeline.sample import CellSample

log = get_logger(__name__)

def export_protein_ring_pairs(
        samples: List[CellSample],
        membrane_img,
        protein_img,
        frangi_img,
        scorer,
        paths,
        condition_label=None,
        source_img=None,
        run_id=None,
        version_id=None
    ):


    crop_dir = paths["crops"]
    csv_path = paths["csv"]

    crop_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    log.debug("Beginning export of protein+ring pairs with feature extraction...")

    for sample in samples:
        if not sample.has_ring():
            log.debug(f"Cell {sample.cell_id}: No ring detected, skipping.")
            continue

        features = extract_ring_features(
            sample.region,
            membrane_img,
            protein_img,
            sample.ring_mask,
            frangi_img,
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

        out_path = crop_dir / f"cell_{sample.cell_id:03d}.png"
        save_ring_crop(protein_img, sample.ring_mask, sample.cell_mask, out_path)

    if not rows:
        log.warning("No protein+ring pairs were exported.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    log.info(f"✅ Exported {len(df)} cells to {csv_path.resolve()}")