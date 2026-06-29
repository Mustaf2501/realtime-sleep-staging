"""The REM-detection model (Weco rewrites this file).

Each 30 s epoch is classified REM / not-REM from the feature matrix built in
features.py. build_model returns a scikit-learn-compatible estimator:

    fit(X, y)            X = per-epoch feature rows, y = 1 for REM
    predict(X)           1 for predicted-REM epochs, one label per row

Training rows are many subjects' nights concatenated. A model that needs per-night
boundaries (a sequence model that resets state between nights) can instead declare
fit(X, y, groups), where groups[i] is the subject index of row i; the harness
passes it when the signature asks for it. predict always gets one subject's night
in chronological order.

Real-time constraint. The detector runs live on a watch, so the prediction for the
epoch ending at t may use only data up to t. predict must score each epoch from
that row and earlier ones. That rules out reading later epochs (bidirectional
nets, attention over the whole night), post-hoc smoothing with future predictions,
whole-sequence normalization, and tuning on the held-out subject. evaluate.py
checks this each fold and scores 0 on a violation; the features are already causal
and the threshold is set at train time.

Deployment. The chosen model runs on a phone (Flutter), one epoch at a time, so a
small model that exports to TFLite or ONNX is preferable. evaluate.py measures only
F1 and causality, not size or latency, so treat this as a guideline.

Anything about the model is open: the estimator, its hyperparameters and tuning,
the REM threshold, class weighting, calibration.
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
