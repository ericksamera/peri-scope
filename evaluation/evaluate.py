# evaluation/evaluate.py
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
from pipeline.logger import get_logger
from pathlib import Path

log = get_logger(__name__)

def evaluate_pairs(csv_path, out_dir=None):
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns or 'score' not in df.columns:
        raise ValueError("CSV must contain 'label' and 'score' columns")

    df = df[df['label'].isin(['good', 'bad'])].dropna(subset=['score'])
    if df.empty:
        raise ValueError("No labeled examples found for evaluation")

    y_true = df['label'].map({'bad': 0, 'good': 1}).values
    y_score = df['score'].values
    y_pred = (y_score >= 0.5).astype(int)

    print("\n🔍 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['bad', 'good'], digits=3))

    try:
        auc = roc_auc_score(y_true, y_score)
        print(f"AUC: {auc:.3f}")
    except Exception as e:
        log.warning(f"AUC computation failed: {e}")

    plot_score_distributions(df, out_dir)
    plot_roc_pr_curves(y_true, y_score, out_dir)

def plot_score_distributions(df, out_dir=None):
    plt.figure(figsize=(6, 4))
    df_good = df[df["label"] == "good"]
    df_bad = df[df["label"] == "bad"]
    plt.hist(df_good["score"], bins=20, alpha=0.5, label="good", color="green")
    plt.hist(df_bad["score"], bins=20, alpha=0.5, label="bad", color="red")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Score Distribution by Label")
    plt.legend()
    plt.tight_layout()
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / "score_distribution.png", dpi=300)
    else:
        plt.show()
    plt.close()

def plot_roc_pr_curves(y_true, y_score, out_dir=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)

    plt.tight_layout()

    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(Path(out_dir) / "roc_pr_curves.png", dpi=300)
    else:
        plt.show()
    plt.close()
