# cli.py
import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from pipeline.utils import load_tiff_channels
from skimage.filters import frangi
from skimage.measure import regionprops
from pipeline.apply_cnn import apply_cnn_model
from pipeline.visualize import overlay_rings
from pipeline.segmentation import segment_cells
from pipeline.ring_detection import extract_rings
from pipeline.train_grid import train_model_with_grid
from pipeline.export import export_protein_ring_pairs
from pipeline.utils import normalize_image, get_next_version_id

from concurrent.futures import ThreadPoolExecutor

from pipeline.logger import get_logger
from pipeline.scorer import RuleBasedScorer, MLScorer
from pipeline.train import train_classifier
from evaluation.evaluate import evaluate_pairs
from config import OUTPUTS_DIR, DEFAULT_SCORER, MIN_CELL_AREA
import logging
from itertools import count

from config import get_output_paths

from pipeline.sample import CellSample
import config

log_file = Path("peri-scope.log")
file_handler = logging.FileHandler(log_file, mode='a')
file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)
root_logger.setLevel(logging.DEBUG)

log = get_logger(__name__)

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


def process_tif(tif_path, label, scorer, run_id, version_id, paths, debug, current_cell_id):
    from shutil import copy2
    from datetime import datetime
    from pathlib import Path

    log.info(f"🔍 Processing {tif_path} [label: {label}]")
    copy2(tif_path, paths["run"] / Path(tif_path).name)

    protein_img, membrane_img = load_tiff_channels(tif_path)
    norm_membrane = normalize_image(membrane_img)
    norm_protein = normalize_image(protein_img)
    frangi_img = frangi(norm_membrane)

    cell_labels = segment_cells(norm_membrane, out_dir=paths["debug_cells"], debug=debug)
    samples = []

    for region in regionprops(cell_labels):
        if region.area >= MIN_CELL_AREA:
            sample = CellSample(next(current_cell_id), region, cell_labels == region.label)
            sample._membrane_img = membrane_img
            sample._norm_protein = norm_protein
            sample._frangi_img = frangi_img
            sample.meta.update({
                "condition": label,
                "source_img": tif_path,
                "run_id": run_id,
                "version_id": version_id,
            })
            samples.append(sample)

    if not samples:
        log.warning(f"No valid cells found in {tif_path}")
        return [], None

    if debug:
        ring_mask = extract_rings(
            membrane_img, samples, norm_protein, scorer, frangi_img,
            debug=True, debug_dir=paths["debug_cells"]
        )
        overlay_rings(
            norm_protein,
            membrane_img,
            cell_labels,
            ring_mask,
            out_dir=paths["run"] / "debug" / Path(tif_path).stem
        )
    else:
        ring_mask = extract_rings(membrane_img, samples, norm_protein, scorer, frangi_img, debug=False)

    export_protein_ring_pairs(
        samples=samples,
        scorer=scorer,
        paths=paths,
        condition_label=label,
        source_img=tif_path,
        run_id=run_id,
        version_id=version_id
    )

    meta = {
        "tif": tif_path,
        "condition": label,
        "source_img": tif_path,
        "scorer_type": scorer.__class__.__name__,
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "version_id": version_id,
    }

    return samples, meta


def apply_config_overrides(override_str):
    if not override_str:
        return

    for pair in override_str.split(","):
        if "=" not in pair:
            continue
        key, val = pair.split("=")
        key = key.strip()
        val = val.strip()

        # Try to infer the correct type
        if hasattr(config, key):
            current = getattr(config, key)
            try:
                if isinstance(current, bool):
                    val = val.lower() in ("1", "true", "yes", "on")
                elif isinstance(current, int):
                    val = int(val)
                elif isinstance(current, float):
                    val = float(val)
                else:
                    val = str(val)
                setattr(config, key, val)
                log.info(f"[override] {key} = {val}")
            except Exception as e:
                log.warning(f"Could not override {key}: {e}")
        else:
            log.warning(f"[override] Unknown config key: {key}")

def cmd_detect_export(args):

    run_id = get_next_version_id(OUTPUTS_DIR, prefix="run")
    run_path = Path(OUTPUTS_DIR) / f"{run_id}"
    version_id = get_next_version_id(run_path / "versions")
    paths = get_output_paths(run_id, version_id)
    run_path.mkdir(parents=True, exist_ok=True)
    paths["version"].mkdir(parents=True, exist_ok=True)

    # Pair TIFFs and labels
    if args.label:
        if len(args.label) != len(args.tif):
            raise ValueError("Number of --label entries must match number of --tif files")
        tif_label_pairs = list(zip(args.tif, args.label))
    else:
        tif_label_pairs = [(tif, args.label) for tif in args.tif]

    # Initialize scorer once
    scorer = load_scorer(args.weights) if args.weights else RuleBasedScorer(
        weights=DEFAULT_SCORER["weights"],
        bias=DEFAULT_SCORER["bias"]
    )

    all_samples = []
    current_cell_id = count(1)
    all_metadata = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                process_tif, tif_path, label, scorer, run_id, version_id, paths, args.debug, current_cell_id
            )
            for tif_path, label in tif_label_pairs
        ]

        all_samples = []
        all_metadata = []
        for fut in futures:
            samples, meta = fut.result()
            if samples:
                all_samples.extend(samples)
            if meta:
                all_metadata.append(meta)


    save_metadata(paths["metadata"], {"inputs": all_metadata})
    save_metadata(paths["source_metadata"], {"inputs": all_metadata})

    log.info(f"✅ Done: run_{run_id} → version {version_id}")


def cmd_train(args):
    train_classifier(args.csv, rescore=args.rescore)

def cmd_train_grid(args):
    train_model_with_grid(args.csv, rescore=args.rescore)

def cmd_evaluate(args):
    evaluate_pairs(args.csv, out_dir=Path(args.csv).parent)

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

    p_detect = subparsers.add_parser("detect-export", help="Detect and export membrane-localized protein rings")
    p_detect.add_argument(
        "--tif",
        action="append",
        required=True,
        help="Path to input TIFF (channel 0 = protein, 1 = membrane). Repeat to pass multiple images."
    )
    p_detect.add_argument(
        "--label",
        action="append",
        required=False,
        help="Condition label(s) per TIFF (must match order of --tif)"
    )
    p_detect.add_argument("--weights", help="Path to scorer (.json or .pkl)")
    p_detect.add_argument("--debug", action="store_true", help="Enable debug overlays and visual outputs")
    p_detect.add_argument("--override", help="Comma-separated config overrides like CELLPOSE_DIAMETER=90")
    p_detect.set_defaults(func=cmd_detect_export)


    p_train = subparsers.add_parser("train", help="Train a new model based on labeled data")
    p_train.add_argument("--csv", required=True, help="Path to labeled pairs_metadata.csv")
    p_train.add_argument("--rescore", action="store_true", help="Immediately apply trained model to create a new version")
    p_train.set_defaults(func=cmd_train)

    p_train_grid = subparsers.add_parser("train-grid", help="Train classifier with grid search")
    p_train_grid.add_argument("--csv", required=True, help="Path to labeled pairs_metadata.csv")
    p_train_grid.add_argument("--rescore", action="store_true", help="Immediately apply best model to create new version")
    p_train_grid.set_defaults(func=cmd_train_grid)
    
    p_eval = subparsers.add_parser("evaluate", help="Evaluate scored pairs")
    p_eval.add_argument("--csv", required=True, help="Path to pairs_metadata.csv with scores")
    #p_eval.add_argument("--out-dir", help="Directory to save evaluation plots (PNG)")
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