# pipeline/ring_detection.py
import numpy as np
from skimage.filters import sobel
from skimage.morphology import binary_erosion, binary_dilation, disk
from config import (
    RING_SCALES,
    RING_DILATE_PX,
    RING_PERCENTILE_CUTOFF,
)
from pipeline.utils import normalize_image
from pipeline.logger import get_logger

from typing import List
from pipeline.sample import CellSample

log = get_logger(__name__)

# def extract_rings(membrane_img, samples: List[CellSample], protein_img=None, scorer=None, frangi_img=None, debug=False, debug_dir=None) -> np.ndarray:
    
#     ring_mask = np.zeros_like(membrane_img, dtype=bool)
#     mem_norm = normalize_image(membrane_img)
#     mem_edges = sobel(mem_norm)

#     for sample in samples:
#         region = sample.region

#         minr, minc, maxr, maxc = region.bbox
#         submask = region.image
#         est_radius = np.sqrt(region.area / np.pi)
#         radii = [max(1, int(scale * est_radius)) for scale in RING_SCALES]

#         ring_union = np.zeros_like(submask, dtype=bool)
#         for r in radii:
#             inner = binary_erosion(submask, disk(r))
#             ring = submask ^ inner
#             ring_union |= ring

#         if not ring_union.any():
#             if debug:
#                 log.debug(f"Cell {sample.cell_id}: No ring after erosion.")
#                 if debug and debug_dir:
#                     sample.save_debug(protein_img, membrane_img, ring_mask, out_dir=debug_dir)
#             continue

#         edge_vals = mem_edges[minr:maxr, minc:maxc][ring_union]
#         score_fraction = np.mean(edge_vals > 0)

#         if score_fraction < 0.5:  # <- this can be a configurable cutoff later
#             if debug:
#                 log.debug(f"Cell {sample.cell_id}: Low edge fraction.")
#                 if debug and debug_dir:
#                     sample.save_debug(protein_img, membrane_img, ring_mask, out_dir=debug_dir)
#             continue

#         cutoff = np.percentile(edge_vals, RING_PERCENTILE_CUTOFF)
#         selected = edge_vals > cutoff
#         coords = np.where(ring_union)
#         selected_y = coords[0][selected]
#         selected_x = coords[1][selected]

#         if len(selected_y) < 15:
#             if debug:
#                 log.debug(f"Cell {sample.cell_id}: Ring too small.")
#                 if debug and debug_dir:
#                     sample.save_debug(protein_img, membrane_img, ring_mask, out_dir=debug_dir)
#             continue

#         # Update the full image ring_mask
#         ring_mask[minr + selected_y, minc + selected_x] = True

#         # Create per-cell ring and attach to sample
#         cell_ring = np.zeros_like(sample.cell_mask, dtype=bool)
#         cell_ring[minr:maxr, minc:maxc][selected_y, selected_x] = True
#         sample.set_ring_mask(cell_ring)

#         if debug:
#             log.debug(f"Cell {sample.cell_id}: ring area = {len(selected_y)}")
#             if debug and debug_dir:
#                 sample.save_debug(protein_img, membrane_img, ring_mask, out_dir=debug_dir)

#     if RING_DILATE_PX > 0:
#         ring_mask = binary_dilation(ring_mask, disk(RING_DILATE_PX))

#     return ring_mask



from config import RING_WIDTH_SCALE, RING_WIDTH_MIN, RING_WIDTH_MAX, RING_DILATE_PX, RING_INTENSITY_CUTOFF

def extract_rings(membrane_img, samples: List[CellSample], protein_img=None, scorer=None, frangi_img=None, debug=False, debug_dir=None) -> np.ndarray:
    ring_mask = np.zeros_like(membrane_img, dtype=bool)
    mem_norm = normalize_image(membrane_img)

    for sample in samples:
        region = sample.region
        minr, minc, maxr, maxc = region.bbox
        submask = region.image

        # Dynamically estimate ring width
        est_radius = np.sqrt(region.area / np.pi)
        ring_width = int(est_radius * RING_WIDTH_SCALE)
        ring_width = max(RING_WIDTH_MIN, min(ring_width, RING_WIDTH_MAX))

        outer = binary_dilation(submask, disk(ring_width))
        inner = binary_erosion(submask, disk(ring_width))
        local_ring = outer ^ inner

        if local_ring.shape != submask.shape:
            log.warning(f"Cell {sample.cell_id}: shape mismatch in ring mask.")
            continue

        if not local_ring.any():
            if debug:
                log.debug(f"Cell {sample.cell_id}: Empty ring mask.")
                sample.save_debug(protein_img, membrane_img, ring_mask, out_dir=debug_dir)
            continue

        mem_crop = mem_norm[minr:maxr, minc:maxc]
        if mem_crop.shape != local_ring.shape:
            log.warning(f"Cell {sample.cell_id}: shape mismatch between membrane crop and ring.")
            continue

        ring_vals = mem_crop[local_ring]
        if ring_vals.size == 0:
            if debug:
                log.debug(f"Cell {sample.cell_id}: No pixels in ring_crop.")
                sample.save_debug(protein_img, membrane_img, ring_mask, out_dir=debug_dir)
            continue

        cutoff = np.percentile(ring_vals, RING_INTENSITY_CUTOFF)
        selected = ring_vals > cutoff
        coords = np.where(local_ring)
        selected_y = coords[0][selected]
        selected_x = coords[1][selected]

        if len(selected_y) < 15:
            if debug:
                log.debug(f"Cell {sample.cell_id}: Ring too small after thresholding.")
                sample.save_debug(protein_img, membrane_img, ring_mask, out_dir=debug_dir)
            continue

        # Update full-image ring mask
        ring_mask[minr + selected_y, minc + selected_x] = True

        # Per-cell ring mask
        cell_ring = np.zeros_like(sample.cell_mask, dtype=bool)
        cell_ring[minr:maxr, minc:maxc][selected_y, selected_x] = True
        sample.set_ring_mask(cell_ring)

        if debug:
            log.debug(f"Cell {sample.cell_id}: ring width = {ring_width}, area = {len(selected_y)}")
            sample.save_debug(protein_img, membrane_img, ring_mask, out_dir=debug_dir)

    if RING_DILATE_PX > 0:
        ring_mask = binary_dilation(ring_mask, disk(RING_DILATE_PX))

    return ring_mask
