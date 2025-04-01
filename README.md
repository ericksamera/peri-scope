# Peri-scope: Protein Ring Detection and Scoring Pipeline

**Peri-scope** is a modular, versioned, and CLI-driven pipeline for detecting membrane-localized protein rings in 2-channel TIFF images, extracting per-cell features, and scoring them using rules, classical ML, or CNNs. It supports detailed visual debugging, versioned outputs, and is designed for reproducibility and inspection.

---

## What It Does

- **Segment cells** from the membrane channel using Cellpose.
- **Detect membrane-localized rings** per cell using erosion + edge detection.
- **Extract features** for each ring from the protein and membrane images.
- **Score rings** with rule-based, ML-based, or CNN-based scorers.
- **Export crops + metadata** per cell for training or QC.
- **Version every step** for reproducible experiments.
- **Visually debug** all steps with optional overlays.

---

## Biological Motivation

This tool supports quantitative analysis of protein localization to membranes — useful when comparing **wild-type vs mutant constructs**. It aims to provide:

- Automated scoring of membrane trafficking.
- Per-cell visual inspection.
- Trainable models using both features and CNNs.

---

## How the Pipeline Works

### Input
- A 2-channel `.tif` file: `channel 0 = protein`, `channel 1 = membrane`

### 1. Detection & Export (`detect-export`)
- Segments cells with Cellpose.
- Detects candidate rings (via erosion + Sobel edge thresholding).
- Computes features per cell (intensity, geometry, enrichment, Frangi).
- Scores each ring using a scorer (rule-based or ML).
- Saves:
  - `pairs_metadata.csv`
  - Cropped ring images
  - Debug overlays if `--debug`
  - Cell/segmentation masks and ring masks

### 2. Labeling
- User adds `label` column (`good` / `bad`) in `pairs_metadata.csv` manually.

### 3. Training
- `train`: trains a RandomForest classifier on labeled CSV
- `train-grid`: does a grid search for best parameters
- With `--rescore`, the model is applied to the original run → creates a new version folder

### 4. CNN Training (Optional)
- `cnn/train_cnn.py`: trains ResNet18 on cropped ring images using labels in CSV

### 5. CNN Application
- `apply-cnn`: applies CNN to crops, rescoring each ring
- Generates labeled overlays and outputs new versioned `pairs_metadata.csv`

### 6. Evaluation
- `evaluate`: generates classification report, AUC, ROC/PR plots

---

## CLI Commands

```bash
# 1. Detect rings and export features
python cli.py detect-export --tif input.tif --label WT --debug

# 2. Train a classical model
python cli.py train --csv outputs/.../pairs_metadata.csv --rescore

# 3. Grid search training
python cli.py train-grid --csv outputs/.../pairs_metadata.csv --rescore

# 4. Evaluate performance
python cli.py evaluate --csv outputs/.../pairs_metadata.csv

# 5. Rescore with new model or rule weights
python cli.py rescore --csv pairs_metadata.csv --weights model.pkl

# 6. Apply CNN-based model
python cli.py apply-cnn --csv outputs/.../pairs_metadata.csv --model cnn_model.pt
```

---

## Output Structure

```text
outputs/run_001/
├── versions/
│   ├── 001/
│   │   ├── pairs_metadata.csv
│   │   ├── crops/
│   │   ├── model.pkl
│   │   ├── train_metadata.json
│   ├── 002/  <-- new version from rescoring
├── debug/
    └── cells/ (if --debug)
```

---

## Core Components

### Segmentation
- Cellpose (`cyto2`) on membrane channel
- Filters small objects

### Ring Detection
- Multi-scale erosion + edge filtering
- Sobel edge + percentile cutoff

### Feature Extraction
- Intensity, geometry, Frangi features
- Dynamic enrichment metrics via plugin system

### Scoring
- RuleBasedScorer (linear combo)
- MLScorer (RandomForest via sklearn)
- CNNScorer (ResNet18, PyTorch)

### Export & Debug
- `pairs_metadata.csv`: scores + features
- `crops/`: ring-centered image crops
- `debug/cells/`: per-cell overlays (optional)

---

## Configuration (config.py)
- Cellpose diameter, model type
- Ring detection thresholds + radii
- Output layout, padding, overlay alpha
- Default rule-based scorer weights

---

## Labeling Workflow
- Open exported `pairs_metadata.csv`
- Add `label` column (`good` / `bad`)
- Use for training ML or CNN models

---

## Future Plans
- GUI for visual labeling
- Embedding viewer for all cells
- Wild-type vs mutant statistical comparison tools
- Advanced CNN scoring with attention/segmentation

    