# evaluate/thresholding.py

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from skimage.filters import threshold_otsu

def suggest_otsu_threshold(scores):
    return threshold_otsu(np.array(scores))

def suggest_best_f1_threshold(scores, labels):
    labels = pd.Series(labels).map({'bad': 0, 'good': 1}).values  # ADD THIS LINE
    best_thresh = 0.5
    best_f1 = 0
    thresholds = np.linspace(0.0, 1.0, 200)

    for t in thresholds:
        preds = [s >= t for s in scores]
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1
