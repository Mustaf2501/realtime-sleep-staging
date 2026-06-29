"""REM-detection model  —  THIS FILE IS OPTIMIZED BY WECO.

Goal: classify each 30 s epoch as REM / not-REM for targeted lucidity
reactivation. Features are fixed and causal in features.py and cannot be changed
here; Weco optimizes only the model that maps the per-epoch feature matrix to a
REM / not-REM decision (any scikit-learn-compatible classifier — random forest,
gradient boosting, SVM, MLP, or an sklearn-wrapped torch/keras model).

================================  CONTRACT  ================================
    build_model() -> a scikit-learn-compatible estimator
        fit(X, y)            trains on stacked per-epoch feature rows (y: 1 == REM)
        predict(X)           -> 1 for predicted-REM epochs, one label per row

    The training rows are many subjects' nights concatenated. A model that needs
    per-night boundaries (a sequence model resetting state between nights) may
    instead declare:
        fit(X, y, groups)    groups[i] = subject index of row i (contiguous per
                             night); the harness passes it automatically when the
                             signature has `groups`. Tabular models just use (X, y).

    predict(X) always receives ONE held-out subject's epochs in chronological
    order, so a causal model can process them as a single stream.

The harness (evaluate.py) cross-validates it leave-one-subject-out and reports the
mean per-subject REM F1.
===========================================================================

----------------------------  REAL-TIME RULE  -----------------------------
This model is meant to run live on a wrist device: at the end of the epoch ending
at time t it must emit that epoch's decision immediately, from data already seen.

    The prediction for the epoch ending at t may depend ONLY on information with
    timestamp <= t. Nothing from a later epoch may influence an earlier
    prediction. Concretely, predict() must score each epoch from that row and
    earlier rows only (a unidirectional / streaming computation).

"Cheating" means breaking that rule by letting the future leak in, e.g.:
  - reading later rows to label the current one (bidirectional RNN/Transformer,
    attention over the whole night);
  - post-hoc smoothing of the prediction sequence with future predictions
    (median filter, full-night Viterbi, "relabel if surrounded by REM");
  - normalizing features by whole-sequence statistics (mean/std/min/max computed
    over all epochs, including future ones);
  - tuning the threshold or any parameter on the held-out subject's data.

evaluate.py ENFORCES this: every fold is checked for prefix-invariance — the
first-k predictions must be unchanged when later epochs are removed or altered.
Any look-ahead scores metric = 0. (Feature-level look-ahead is already impossible:
features.py is fixed and causal, and the threshold is fixed at train time.)
---------------------------------------------------------------------------

--------------------------  DEPLOYMENT (mobile)  --------------------------
The winning model ships in a Flutter phone app and runs on-device, one epoch at a
time, so prefer models that are:
  - small and low-latency (per-epoch inference is tiny compute);
  - exportable to a mobile runtime — TFLite / ONNX (sklearn trees convert via
    skl2onnx; torch/keras via their exporters) — avoid exotic/unsupported ops;
  - deterministic at inference (eval mode, no dropout) so the causality check is
    stable and on-device output is reproducible.
These are guidance, not gates: evaluate.py scores only REM F1 + causality, so it
will not by itself penalize a large or un-exportable model — keep them in mind
(or steer Weco with `-i`) when choosing a winner to deploy.
---------------------------------------------------------------------------

Free to change: the estimator, its hyperparameters, hyperparameter tuning methods, the REM probability
threshold, class weighting, calibration — anything about the model, as long as
the contract and the real-time rule hold.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import FixedThresholdClassifier

REM_THRESHOLD = 0.24      # P(REM) >= threshold -> REM  (paper used 0.24)
RF_KWARGS = dict(n_estimators=200, min_samples_leaf=48, n_jobs=-1, random_state=0)


def build_model():
    """The classifier the harness cross-validates: a random forest whose
    .predict() applies the REM probability threshold (via FixedThresholdClassifier)
    instead of the default 0.5."""
    forest = RandomForestClassifier(**RF_KWARGS)
    return FixedThresholdClassifier(
        forest, threshold=REM_THRESHOLD, pos_label=1, response_method="predict_proba")
