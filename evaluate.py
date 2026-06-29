"""Leave-one-subject-out scoring for the model in module.py.

Prints `metric: <value>` for Weco to maximize: the mean per-subject REM F1 (F1 on
each held-out subject, averaged across folds). Run directly for the full breakdown.

Scoring follows the paper's Figure 1. Accuracy, precision, recall, and F1 are
computed per held-out subject and averaged across folds (mean ± SEM); a subject
with no scored REM is skipped, since REM precision and recall are undefined there.
The confusion matrix is pooled over all epochs and row-normalized (the paper
row-normalizes the matrix but averages the metrics across folds), so its
[REM, REM] cell is close to, but not exactly, the averaged recall.

Each run writes results/<model-hash>.{py,json,png}, keyed by a hash of module.py:
the .py is the exact model, the .json the numbers, the .png the figure. Weco logs
the metric and the code itself; these are extra.

Before scoring, each fold is checked for look-ahead (the real-time constraint in
module.py): if the first-k predictions change when later epochs are removed or
altered, the model is reading the future and scores 0.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

import splits
from module import build_model

REM_LABEL = 1
_CUT_FRACTIONS = (0.25, 0.5, 0.75)   # points in the night where look-ahead is checked
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
_LABELS = ["Wake/NREM", "REM"]


def _fit(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    """Fit the model, handing per-night subject boundaries to models that ask for
    them. A model whose fit signature declares `groups` (e.g. a sequence model
    that resets state between nights) receives groups[i] = the subject index of
    training row i; a plain tabular estimator just gets (X, y)."""
    if "groups" in inspect.signature(model.fit).parameters:
        return model.fit(X, y, groups=groups)
    return model.fit(X, y)


def _predictions_are_causal(model, X_test: np.ndarray, full_pred: np.ndarray) -> bool:
    """True if the model scores each epoch using only that epoch and earlier ones.

    At several cut points k we require the first-k predictions to be identical
    (a) when epochs after k are removed, and (b) when their content is zeroed.
    A model that peeks at the future fails at least one of these.
    """
    n = len(X_test)
    for frac in _CUT_FRACTIONS:
        k = max(1, int(n * frac))
        if not np.array_equal(full_pred[:k], model.predict(X_test[:k])[:k]):
            return False
        altered_future = X_test.copy()
        altered_future[k:] = 0.0
        if not np.array_equal(full_pred[:k], model.predict(altered_future)[:k]):
            return False
    return True


def _mean_sem(values: list[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return float(a.mean()), sem


def main() -> float:
    # features (X): (n_epochs, n_features) | labels (y): (n_epochs,), 1 == REM
    # subjects (groups): (n_epochs,)   -- from the committed matrix when present
    X, y, groups = splits.load_dataset()

    per_fold = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    pooled_true, pooled_pred = [], []   # every epoch, for the pooled confusion (A)
    skipped = 0                         # subjects with no scored REM
    for train_idx, test_idx in splits.cross_validator().split(X, y, groups=groups):
        model = _fit(build_model(), X[train_idx], y[train_idx], groups[train_idx])
        X_test, y_test = X[test_idx], y[test_idx]
        y_pred = model.predict(X_test)

        if not _predictions_are_causal(model, X_test, y_pred):
            print("metric: 0.0")
            print(f"CAUSALITY CHECK FAILED on subject group {int(groups[test_idx][0])}: "
                  "predictions for earlier epochs change when later epochs are removed "
                  "or altered — the model looks ahead and is not real-time.")
            return 0.0

        pooled_true.append(y_test)             # the pooled confusion uses every subject
        pooled_pred.append(y_pred)

        if (y_test == REM_LABEL).sum() == 0:   # no REM -> REM metrics undefined, skip (B)
            skipped += 1
            continue

        per_fold["accuracy"].append(accuracy_score(y_test, y_pred))
        per_fold["precision"].append(
            precision_score(y_test, y_pred, pos_label=REM_LABEL, zero_division=0))
        per_fold["recall"].append(
            recall_score(y_test, y_pred, pos_label=REM_LABEL, zero_division=0))
        per_fold["f1"].append(
            f1_score(y_test, y_pred, pos_label=REM_LABEL, zero_division=0))

    stats = {name: _mean_sem(vals) for name, vals in per_fold.items()}
    confusion = _row_normalized_confusion(          # pooled over all epochs (paper's A)
        np.concatenate(pooled_true), np.concatenate(pooled_pred))
    n_subjects = len(per_fold["f1"])

    f1_mean = stats["f1"][0]
    print(f"metric: {f1_mean:.4f}")
    for name in ("f1", "accuracy", "precision", "recall"):
        mean, sem = stats[name]
        print(f"{name}: {mean:.4f} +/- {sem:.4f} SEM")
    note = f" ({skipped} skipped: no scored REM)" if skipped else ""
    print(f"(per-subject mean over {n_subjects} folds{note})")

    _save_results(stats, confusion, n_subjects)
    return f1_mean


def _row_normalized_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=[0, REM_LABEL]).astype(float)
    row_totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_totals, out=np.full_like(cm, np.nan), where=row_totals > 0)


def _model_hash() -> str:
    import module
    return hashlib.sha256(open(module.__file__, "rb").read()).hexdigest()[:12]


def _save_results(stats: dict, confusion: np.ndarray, n_subjects: int) -> None:
    """Write results/<model-hash>.{json,png} — numbers always, figure too."""
    import module
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_hash = _model_hash()
    base = os.path.join(RESULTS_DIR, model_hash)

    with open(base + ".py", "w") as f:          # the exact model — makes the hash identifiable
        f.write(open(module.__file__).read())

    with open(base + ".json", "w") as f:
        json.dump({
            "model_hash": model_hash,
            "metric_mean_rem_f1": stats["f1"][0],
            "n_subjects": n_subjects,
            "per_subject": {k: {"mean": m, "sem": s} for k, (m, s) in stats.items()},
            "confusion_rownorm": confusion.tolist(),
            "confusion_labels": _LABELS,
            "confusion_method": "pooled, row-normalized over all epochs",
        }, f, indent=2)

    _save_figure(base + ".png", stats, confusion, model_hash)


def _save_figure(path: str, stats: dict, confusion: np.ndarray, model_hash: str) -> None:
    import matplotlib
    matplotlib.use("Agg")   # headless: no display needed
    import matplotlib.pyplot as plt

    fig, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

    ax_cm.imshow(confusion, cmap="Blues", vmin=0, vmax=1)
    ax_cm.set_xticks([0, 1], _LABELS)
    ax_cm.set_yticks([0, 1], _LABELS)
    for i in range(2):
        for j in range(2):
            ax_cm.text(j, i, f"{confusion[i, j]:.2f}", ha="center", va="center")
    ax_cm.set_xlabel("Predicted sleep stage")
    ax_cm.set_ylabel("True sleep stage")
    ax_cm.set_title("Confusion (row-normalized, avg over folds)")

    names = ["Accuracy", "Precision", "Recall"]
    means = [stats["accuracy"][0], stats["precision"][0], stats["recall"][0]]
    sems = [stats["accuracy"][1], stats["precision"][1], stats["recall"][1]]
    ax_bar.bar(names, means, yerr=sems, capsize=4, color="0.8", edgecolor="black")
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Ratio")
    ax_bar.set_title("Per-fold mean +/- SEM")

    fig.suptitle(f"REM detection — model {model_hash}  (REM F1 = {stats['f1'][0]:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
