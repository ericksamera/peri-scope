# pipeline/segmentation.py
import numpy as np
from skimage.morphology import remove_small_objects
from cellpose import models
from config import CELLPOSE_MODEL_TYPE, CELLPOSE_DIAMETER, CELLPOSE_MIN_AREA
from pipeline.logger import get_logger

log = get_logger(__name__)

def segment_cells(img: np.ndarray, out_dir="debug_segmentation", debug=False) -> np.ndarray:
    # Initialize Cellpose model
    model = models.Cellpose(model_type=CELLPOSE_MODEL_TYPE)
    
    # Perform segmentation with Cellpose
    masks, _, _, _ = model.eval(img, diameter=CELLPOSE_DIAMETER, channels=[0, 0])
    
    # Clean small objects
    cleaned = remove_small_objects(masks.astype(np.int32), min_size=CELLPOSE_MIN_AREA)

    if debug:
        for cell_id in np.unique(cleaned):
            if cell_id == 0: continue  # Skip background

    return cleaned
