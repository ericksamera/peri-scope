# config.py
from pathlib import Path

# Segmentation
CELLPOSE_MODEL_TYPE = "cyto2"
CELLPOSE_DIAMETER = 70
CELLPOSE_MIN_AREA = 50

# Ring detection
RING_SCALES = (0.1, 0.4)
RING_DILATE_PX = 1
RING_PERCENTILE_CUTOFF = 5.0
MIN_CELL_AREA = 10

# Default rule-based scorer weights
DEFAULT_SCORER = {
    "weights": {
        "protein_mean": 1.0,
        "frangi_mean": 0.5,
        "solidity": 0.2,
        "eccentricity": -0.1
    },
    "bias": -0.2
}

# Output
OUTPUTS_DIR = "outputs"
CROP_PADDING = 100
RING_OVERLAY_ALPHA = 80

def get_output_paths(run_id: str, version_id: str):
    run_base = Path(OUTPUTS_DIR) / f"{run_id}"
    version_base = run_base / "versions" / f"{version_id}"
    return {
        "run": run_base,
        "version": version_base,
        "debug_cells": run_base / "debug/cells",
        "overlay": run_base / "debug",
        "metadata": run_base / "metadata.json",
        "source_metadata": version_base / "source_metadata.json",
        "crops": version_base / "crops",
        "csv": version_base / "pairs_metadata.csv",
    }
