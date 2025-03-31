# pipeline/feature_extraction.py
import numpy as np
from pipeline.logger import get_logger
from skimage.morphology import binary_erosion, binary_dilation, disk

from pipeline.metrics_registry import compute_registered_metrics

from typing import List
from pipeline.sample import CellSample

log = get_logger(__name__)

def extract_features_for_samples(samples: List[CellSample], membrane_img: np.ndarray, protein_img: np.ndarray, frangi_img: np.ndarray = None ):
    for sample in samples:
        if not sample.has_ring():
            log.debug(f"Cell {sample.cell_id}: No ring — skipping feature extraction.")
            continue

        features = extract_ring_features(
            sample.region,
            membrane_img,
            protein_img,
            sample.ring_mask,
            frangi_img,
        )

        if features:
            sample.set_features(features)
        else:
            log.debug(f"Cell {sample.cell_id}: Feature extraction failed.")

def extract_ring_features(region, membrane_img, protein_img, ring_mask, frangi_img=None):
    minr, minc, maxr, maxc = region.bbox
    ring = ring_mask[minr:maxr, minc:maxc]
    cell_mask = region.image

    if not np.any(ring):
        return {}

    features = {}

    # Core intensity values
    protein_crop = protein_img[minr:maxr, minc:maxc]
    ring_vals = protein_crop[ring]
    cell_vals = protein_crop[cell_mask]

    features["protein_mean"] = np.mean(ring_vals)
    features["protein_std"] = np.std(ring_vals)

    # Frangi features
    if frangi_img is not None:
        frangi_vals = frangi_img[minr:maxr, minc:maxc][ring]
        features["frangi_mean"] = np.mean(frangi_vals)
        features["frangi_std"] = np.std(frangi_vals)

    # Shape
    features["area"] = region.area
    features["eccentricity"] = region.eccentricity
    features["solidity"] = region.solidity
    features["ring_area"] = float(np.sum(ring))

    # Create inner and outer ring masks for enrichment calculations
    inner = binary_erosion(cell_mask, disk(3))
    outer = binary_dilation(cell_mask, disk(3)) ^ cell_mask

    inner_vals = protein_crop[inner]
    outer_vals = protein_crop[outer]

    features["ring_inner_diff"] = features["protein_mean"] - np.mean(inner_vals) if inner_vals.size else 0
    features["ring_inner_ratio"] = features["protein_mean"] / (np.mean(inner_vals) + 1e-6)
    features["ring_outer_diff"] = features["protein_mean"] - np.mean(outer_vals) if outer_vals.size else 0
    features["ring_outer_ratio"] = features["protein_mean"] / (np.mean(outer_vals) + 1e-6)

    # Add dynamic metrics from registry
    dynamic = compute_registered_metrics(region, protein_crop, ring, inner, outer)
    features.update(dynamic)

    return features