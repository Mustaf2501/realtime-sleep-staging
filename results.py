"""Recording one model's leave-one-subject-out results to disk.

evaluate.py computes the numbers; this module records them. Given the scored stats,
it writes results/<model-hash>.{py,json,png} -- the exact model, the metrics, and
the figure -- keeping file IO, JSON, hashing, and matplotlib out of the scoring code.

Public surface:
    row_normalized_confusion(y_true, y_pred, stages) -> (k, k) array
    save(stats, confusion, per_class, n_subjects, labels, beta) -> writes the files
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def row_normalized_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                             stages: list[int]) -> np.ndarray:
    """Confusion matrix over `stages`, each true-class row normalized to sum 1. A
    stage that never truly occurs leaves its row NaN rather than dividing by zero."""
    cm = confusion_matrix(y_true, y_pred, labels=stages).astype(float)
    row_totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_totals, out=np.full_like(cm, np.nan), where=row_totals > 0)


def save(stats: dict, confusion: np.ndarray, per_class: dict, n_subjects: int,
         labels: list[str], beta: float) -> None:
    """Write results/<model-hash>.{py,json,png} — the model, the numbers, the figure.
    The hash keys on module.py, so identical models overwrite the same files."""
    import module
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_hash = _model_hash()
    base = os.path.join(RESULTS_DIR, model_hash)

    with open(base + ".py", "w") as f:          # the exact model — makes the hash identifiable
        f.write(open(module.__file__).read())

    with open(base + ".json", "w") as f:
        json.dump({
            "model_hash": model_hash,
            "metric_mean_rem_fbeta": stats["fbeta"][0],
            "beta": beta,
            "n_subjects": n_subjects,
            # REM one-vs-rest, per-subject mean/sem; accuracy is overall (all 5 stages)
            "rem_per_subject": {k: {"mean": m, "sem": s} for k, (m, s) in stats.items()},
            # overall multiclass breakdown, pooled over every epoch
            "per_class_pooled": {lbl: {"precision": per_class[lbl]["precision"],
                                       "recall": per_class[lbl]["recall"],
                                       "f1": per_class[lbl]["f1-score"],
                                       "support": int(per_class[lbl]["support"])}
                                 for lbl in labels},
            "overall_accuracy_pooled": per_class["accuracy"],
            "confusion_rownorm": confusion.tolist(),
            "confusion_labels": labels,
            "confusion_method": "pooled, row-normalized over all epochs",
        }, f, indent=2)

    _save_figure(base + ".png", stats, confusion, labels, beta, model_hash)


def _model_hash() -> str:
    import module
    return hashlib.sha256(open(module.__file__, "rb").read()).hexdigest()[:12]


def _save_figure(path: str, stats: dict, confusion: np.ndarray,
                 labels: list[str], beta: float, model_hash: str) -> None:
    import matplotlib
    matplotlib.use("Agg")   # headless: no display needed
    import matplotlib.pyplot as plt

    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

    n = len(labels)
    ax_cm.imshow(confusion, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_xticks(range(n), labels)
    ax_cm.set_yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            val = confusion[i, j]
            ax_cm.text(j, i, "" if np.isnan(val) else f"{val:.2f}",
                       ha="center", va="center", fontsize=8)
    ax_cm.set_xlabel("Predicted sleep stage")
    ax_cm.set_ylabel("True sleep stage")
    ax_cm.set_title("Confusion (row-normalized, pooled)")

    names = ["Accuracy\n(overall)", "REM\nprecision", "REM\nrecall"]
    means = [stats["accuracy"][0], stats["precision"][0], stats["recall"][0]]
    sems = [stats["accuracy"][1], stats["precision"][1], stats["recall"][1]]
    ax_bar.bar(names, means, yerr=sems, capsize=4, color="0.8", edgecolor="black")
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Ratio")
    ax_bar.set_title("Per-fold mean +/- SEM")

    fig.suptitle(f"REM detection - model {model_hash}  (REM F{beta} = {stats['fbeta'][0]:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
