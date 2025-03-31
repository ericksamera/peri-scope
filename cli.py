# cli.py
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image
from skimage.filters import frangi
from skimage.measure import regionprops
from pipeline.apply_cnn import apply_cnn_model
from pipeline.visualize import overlay_rings
from pipeline.segmentation import segment_cells
from pipeline.ring_detection import extract_rings
from pipeline.train_grid import train_model_with_grid
from pipeline.export import export_protein_ring_pairs
from pipeline.utils import normalize_image
from pipeline.logger import get_logger
from pipeline.scorer import RuleBasedScorer, MLScorer
from pipeline.train import train_classifier
from evaluation.evaluate import evaluate_pairs
from config import OUTPUTS_DIR, DEFAULT_SCORER, MIN_CELL_AREA
import logging

from config import get_output_paths

from pipeline.sample import CellSample

import re

log_file = Path("peri-scope.log")
file_handler = logging.FileHandler(log_file, mode='a')
file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.DEBUG)

log = get_logger(__name__)

def load_tiff_channels(tif_path):
    log.debug(f"Loading TIFF: {tif_path}")
    img = Image.open(tif_path)
    img.seek(0)
    protein = np.array(img.copy())
    img.seek(1)
    membrane = np.array(img.copy())
    return protein, membrane

def get_next_version_id(root_dir, prefix="version"):
    """
    Get the next version ID like '001' by scanning subdirectories matching 'prefix_###'
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    version_re = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    ids = []

    for entry in root_dir.iterdir():
        if entry.is_dir():
            match = version_re.match(entry.name)
            if match:
                ids.append(int(match.group(1)))
            else:
                log.debug(f"Skipping non-matching entry: {entry.name}")

    next_id = max(ids) + 1 if ids else 1
    return f"{next_id:03d}"

def save_metadata(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_scorer(path):
    if path.endswith(".json"):
        return RuleBasedScorer.load(path)
    elif path.endswith(".pkl"):
        return MLScorer.load(path)
    else:
        raise ValueError("Unsupported scorer format. Use .json or .pkl")

def cmd_detect_export(args):
    run_id = get_next_version_id(OUTPUTS_DIR, prefix="run")
    run_path = Path(OUTPUTS_DIR) / f"{run_id}"
    version_id = get_next_version_id(run_path / "versions")

    paths = get_output_paths(run_id, version_id)
    run_path.mkdir(parents=True, exist_ok=True)
    paths["version"].mkdir(parents=True, exist_ok=True)

    protein_img, membrane_img = load_tiff_channels(args.tif)
    norm_membrane = normalize_image(membrane_img)
    norm_protein = normalize_image(protein_img)
    frangi_img = frangi(norm_membrane)

    # Perform segmentation and save debug for each cell
    cell_labels = segment_cells(norm_membrane, out_dir=paths["debug_cells"], debug=args.debug)
    samples = [
        CellSample(region.label, region, cell_labels == region.label)
        for region in regionprops(cell_labels)
        if region.area >= MIN_CELL_AREA
    ]

    if args.weights:
        scorer = load_scorer(args.weights)
    else:
        scorer = RuleBasedScorer(weights=DEFAULT_SCORER["weights"], bias=DEFAULT_SCORER["bias"])
    
    debug_dir = paths["debug_cells"] if args.debug else None
    ring_mask = extract_rings(membrane_img, samples, norm_protein, scorer, frangi_img, debug=args.debug, debug_dir=debug_dir)

    # Saving intermediate results for future analysis
    np.save(run_path / "cell_labels.npy", cell_labels)
    np.save(run_path / "ring_mask.npy", ring_mask)
    np.save(run_path / "protein_img.npy", norm_protein)
    np.save(run_path / "membrane_img.npy", membrane_img)

    metadata = {
        "tif": args.tif,
        "scorer_type": scorer.__class__.__name__,
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "version_id": version_id
    }
    save_metadata(paths["metadata"], metadata)
    save_metadata(paths["source_metadata"], metadata)

    # Export the results
    export_protein_ring_pairs(
        samples,
        membrane_img,
        norm_protein,
        frangi_img,
        scorer,
        paths=paths,
        condition_label=args.label,
        source_img=args.tif,
        run_id=run_id,
        version_id=version_id
    )


    # Overlay the protein + membrane + rings (visual debug)
    if args.debug:
        overlay_rings(norm_protein, membrane_img, cell_labels, ring_mask, out_dir=run_path / "debug")

    log.info(f"✅ Done: run_{run_id} → version {version_id}")

def cmd_train(args):
    train_classifier(args.csv, rescore=args.rescore)

def cmd_train_grid(args):
    train_model_with_grid(args.csv, rescore=args.rescore)

def cmd_evaluate(args):
    evaluate_pairs(args.csv)

def cmd_rescore(args):
    scorer = load_scorer(args.weights)
    df = pd.read_csv(args.csv)
    feature_cols = [c for c in df.columns if c not in ("label", "score", "cell_label", "bbox_minr", "bbox_minc", "bbox_maxr", "bbox_maxc")]

    new_scores = []
    for _, row in df.iterrows():
        features = {k: row[k] for k in feature_cols if pd.notna(row[k])}
        score = scorer.score(features)
        new_scores.append(score)

    if args.backup:
        backup_path = Path(args.csv).with_suffix(".bak.csv")
        df.to_csv(backup_path, index=False)
        log.info(f"Original file backed up to {backup_path}")

    df["score"] = new_scores
    df.to_csv(args.csv, index=False)
    log.info(f"Updated scores written to {args.csv}")

def cli():
    parser = argparse.ArgumentParser(description="peri-scope CLI")
    subparsers = parser.add_subparsers(dest="command")

    p_detect = subparsers.add_parser("detect-export")
    p_detect.add_argument("--tif", required=True)
    p_detect.add_argument("--label", help="Condition label (e.g., WT, mutant) to include in export")
    p_detect.add_argument("--weights", help="Path to scorer (.json or .pkl)")
    p_detect.add_argument("--debug", action="store_true", help="Enable debug overlays and visual outputs")
    p_detect.set_defaults(func=cmd_detect_export)

    p_train = subparsers.add_parser("train", help="Train a new model based on labeled data")
    p_train.add_argument("--csv", required=True, help="Path to labeled pairs_metadata.csv")
    p_train.add_argument("--rescore", action="store_true", help="Immediately apply trained model to create a new version")
    p_train.set_defaults(func=cmd_train)

    p_train_grid = subparsers.add_parser("train-grid", help="Train classifier with grid search")
    p_train_grid.add_argument("--csv", required=True, help="Path to labeled pairs_metadata.csv")
    p_train_grid.add_argument("--rescore", action="store_true", help="Immediately apply best model to create new version")
    p_train_grid.set_defaults(func=cmd_train_grid)
    
    p_eval = subparsers.add_parser("evaluate")
    p_eval.add_argument("--csv", required=True, help="Labeled pairs_metadata.csv")
    p_eval.set_defaults(func=cmd_evaluate)

    p_rescore = subparsers.add_parser("rescore")
    p_rescore.add_argument("--csv", required=True, help="pairs_metadata.csv to update")
    p_rescore.add_argument("--weights", required=True, help="New scorer weights (.json or .pkl)")
    p_rescore.add_argument("--backup", action="store_true", help="Backup original file")
    p_rescore.set_defaults(func=cmd_rescore)

    p_apply_cnn = subparsers.add_parser("apply-cnn", help="Apply trained CNN model to ring crops and create new version")
    p_apply_cnn.add_argument("--csv", required=True, help="Path to pairs_metadata.csv")
    p_apply_cnn.add_argument("--model", required=True, help="Path to trained CNN model (.pt)")
    p_apply_cnn.add_argument("--threshold", type=float, default=0.5, help="Threshold for good/bad classification")
    p_apply_cnn.add_argument("--device", default="cpu", help="Device to run model (e.g. cpu, cuda)")
    p_apply_cnn.add_argument("--no-eval", dest="evaluate", action="store_false", help="Skip evaluation after scoring")
    p_apply_cnn.set_defaults(func=lambda args: apply_cnn_model(args.csv, args.model, args.device, args.threshold, args.evaluate))


    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    cli()