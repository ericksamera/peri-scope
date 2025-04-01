# pipeline/metrics_registry.py

import numpy as np
from skimage.morphology import binary_dilation, disk
from pipeline.logger import get_logger

log = get_logger(__name__)

_ring_metric_registry = {}

def ring_metric(func):
    _ring_metric_registry[func.__name__] = func
    return func

def compute_registered_metrics(region, protein_img, ring_mask, inner_mask, outer_mask):
    metrics = {}
    for name, func in _ring_metric_registry.items():
        try:
            metrics[name] = float(func(region, protein_img, ring_mask, inner_mask, outer_mask))
        except Exception as e:
            log.warning(f"Metric '{name}' failed: {e}")
    return metrics

@ring_metric
def ring_total_fraction(region, protein_crop, ring, *_):
    """Total protein in ring vs entire cell."""
    ring_sum = protein_crop[ring].sum()
    cell_sum = protein_crop[region.image].sum()
    return ring_sum / (cell_sum + 1e-6)

@ring_metric
def ring_contrast(region, protein_crop, ring, inner, *_):
    """Normalized contrast of ring vs cytoplasm."""
    ring_mean = protein_crop[ring].mean()
    inner_mean = protein_crop[inner].mean() if inner.any() else 0
    return (ring_mean - inner_mean) / (ring_mean + inner_mean + 1e-6)

@ring_metric
def ring_cv(region, protein_crop, ring, *_):
    """Coefficient of variation in the ring region."""
    vals = protein_crop[ring]
    return np.std(vals) / (np.mean(vals) + 1e-6)

@ring_metric
def membrane_vs_cytoplasm(region, protein_img, ring, *_):
    cell_mask = region.image  # whole cell
    membrane = ring           # ring mask represents membrane
    cytoplasm = cell_mask.copy()
    cytoplasm[membrane] = False  # exclude membrane to get cytoplasm

    mem_intensity = protein_img[membrane].mean()
    cyt_intensity = protein_img[cytoplasm].mean()

    return mem_intensity / (cyt_intensity + 1e-6)
