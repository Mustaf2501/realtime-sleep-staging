"""Leave-one-subject-out scoring for the model in module.py.

Prints `metric: <value>` for Weco to maximize: the mean per-subject REM F-beta
(beta=0.3, which weights precision over recall), computed per held-out subject and
averaged across folds. Run directly for the full breakdown.

Scoring follows the paper's Figure 1. Accuracy, precision, recall, and F-beta are
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

import inspect
import os
import signal

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, fbeta_score,
                             precision_score, recall_score)

import results
import splits
from dataset import N1, N2, N3, REM, WAKE
from module import build_model

STAGES = [WAKE, N1, N2, N3, REM]     # canonical class order for the confusion / report
REM_LABEL = REM                      # the one class the metric is about (multiclass now)
BETA = 0.3                           # F-beta < 1 weights precision over recall
_CUT_FRACTIONS = (0.25, 0.5, 0.75)   # points in the night where look-ahead is checked
_LABELS = ["Wake", "N1", "N2", "N3", "REM"]
EVAL_TIMEOUT_S = 600                 # a candidate that takes longer scores 0 (keeps the search moving)


def _start_watchdog() -> None:
    """Score 0 and exit if the eval runs too long, so a slow/hung candidate never
    stalls the search. (Best-effort: fires between Python operations; a pure
    C-level hang may not be interruptible.)"""
    def _on_timeout(signum, frame):
        print("metric: 0.0")
        print(f"EVAL TIMEOUT: exceeded {EVAL_TIMEOUT_S}s; candidate too slow.")
        os._exit(0)
    signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(EVAL_TIMEOUT_S)


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
    _start_watchdog()                # cap eval wall-clock so no candidate stalls the run
    # features (X): (n_epochs, n_features) | labels (y): (n_epochs,), stage 0..4 (4 == REM)
    # subjects (groups): (n_epochs,)   -- from the committed matrix when present
    X, y, groups = splits.load_dataset()

    per_fold = {"accuracy": [], "precision": [], "recall": [], "fbeta": []}
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

        # REM is one class among five; score it one-vs-rest via labels=[REM].
        per_fold["accuracy"].append(accuracy_score(y_test, y_pred))   # overall, all 5 stages
        per_fold["precision"].append(precision_score(
            y_test, y_pred, labels=[REM_LABEL], average="macro", zero_division=0))
        per_fold["recall"].append(recall_score(
            y_test, y_pred, labels=[REM_LABEL], average="macro", zero_division=0))
        per_fold["fbeta"].append(fbeta_score(
            y_test, y_pred, beta=BETA, labels=[REM_LABEL], average="macro", zero_division=0))

    pooled_true_all = np.concatenate(pooled_true)   # every epoch, every subject
    pooled_pred_all = np.concatenate(pooled_pred)
    stats = {name: _mean_sem(vals) for name, vals in per_fold.items()}
    confusion = results.row_normalized_confusion(     # 5x5, pooled over all epochs
        pooled_true_all, pooled_pred_all, STAGES)
    per_class = classification_report(                # overall, per-stage (pooled)
        pooled_true_all, pooled_pred_all, labels=STAGES, target_names=_LABELS,
        output_dict=True, zero_division=0)
    n_subjects = len(per_fold["fbeta"])

    fbeta_mean = stats["fbeta"][0]
    print(f"metric: {fbeta_mean:.4f}")               # REM F-beta -- what Weco maximizes
    print("REM, per-subject mean +/- SEM:")
    for name in ("fbeta", "precision", "recall"):
        mean, sem = stats[name]
        print(f"  {name}: {mean:.4f} +/- {sem:.4f}")
    acc_mean, acc_sem = stats["accuracy"]
    note = f" ({skipped} skipped: no scored REM)" if skipped else ""
    print(f"overall accuracy: {acc_mean:.4f} +/- {acc_sem:.4f} SEM  "
          f"(per-subject mean over {n_subjects} folds{note}; beta={BETA})")
    print("per class, pooled over all epochs:")
    for lbl in _LABELS:                              # so the other stages are visible too
        r = per_class[lbl]
        print(f"  {lbl:5s} precision {r['precision']:.3f}  recall {r['recall']:.3f}  "
              f"f1 {r['f1-score']:.3f}  (n={int(r['support'])})")

    results.save(stats, confusion, per_class, n_subjects, _LABELS, BETA)
    return fbeta_mean


if __name__ == "__main__":
    main()
