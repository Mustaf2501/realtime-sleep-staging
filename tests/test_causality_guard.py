"""Readable proof that the real-time causality guard works (evaluate.py).

The guard `_predictions_are_causal` must:
  - PASS a model that scores each epoch from its own row only (real-time-safe)
  - FAIL a model that reads a later epoch to decide the current one (cheating)

    uv run --extra test python -m pytest tests/test_causality_guard.py -v
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import _predictions_are_causal


class PerEpochModel:
    """Real-time-safe: each epoch's label depends only on its own feature row."""
    def predict(self, X):
        return (X[:, 0] > 0.5).astype(int)


class LookAheadModel:
    """Cheating: epoch i's label is decided by epoch i+1 (peeks at the future)."""
    def predict(self, X):
        next_row = np.r_[X[1:, 0], X[-1, 0]]
        return (next_row > 0.5).astype(int)


def _example_features(n=40):
    rng = np.random.default_rng(0)
    return rng.random((n, 3))


def test_guard_passes_a_real_time_model():
    X = _example_features()
    model = PerEpochModel()
    assert _predictions_are_causal(model, X, model.predict(X)) is True


def test_guard_catches_a_look_ahead_model():
    X = _example_features()
    model = LookAheadModel()
    assert _predictions_are_causal(model, X, model.predict(X)) is False
