"""The REM-detection model (Weco rewrites this file).

Each 30 s epoch is classified into one of five sleep stages (Wake, N1, N2, N3, REM)
from the feature matrix built in features.py. REM is the class of interest, but the
model sees all five so it can separate REM from each non-REM stage instead of one
collapsed "not-REM" blob. build_model returns a scikit-learn-compatible estimator:

    fit(X, y)            X = per-epoch feature rows, y = stage code 0..4 (4 == REM)
    predict(X)           one predicted stage code per row (4 == REM)

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
REM F-beta and causality, not size or latency, so treat this as a guideline.

Anything about the model is open: the estimator, its hyperparameters and tuning,
the REM threshold, class weighting, calibration.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from dataset import REM

REM_THRESHOLD = 0.24      # P(REM) >= threshold -> call REM  (paper used 0.24)
RF_KWARGS = dict(n_estimators=200, min_samples_leaf=48, n_jobs=-1, random_state=0)


class RemModel:
    """Multiclass stage classifier with a REM bias. predict_proba gives per-stage
    probabilities; predict returns the argmax stage, but overrides to REM whenever
    P(REM) clears a threshold — so the REM precision/recall trade stays tunable even
    though the model is multiclass. Apply the threshold here, in predict — do NOT
    wrap a model in sklearn's FixedThresholdClassifier: that breaks on custom or
    groups-aware estimators. Keep this shape (fit / predict / predict_proba, fit
    accepts groups) for any estimator and they compose cleanly."""

    def __init__(self, threshold: float = REM_THRESHOLD):
        self.threshold = threshold

    def fit(self, X, y, groups=None):     # groups is optional; ignore it if unused
        self.model_ = RandomForestClassifier(**RF_KWARGS).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        proba = self.model_.predict_proba(X)
        pred = self.classes_[np.argmax(proba, axis=1)]
        rem_col = np.flatnonzero(self.classes_ == REM)
        if rem_col.size:                                   # bias toward REM when it clears the threshold
            pred[proba[:, rem_col[0]] >= self.threshold] = REM
        return pred


def build_model():
    return RemModel()
