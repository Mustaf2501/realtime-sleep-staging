"""Tests that the fit contract passes per-night `groups` correctly.

A model whose fit declares `groups` (a sequence model) should receive them; a plain
tabular fit(X, y) should keep working untouched.

    uv run --extra test python -m pytest tests/test_fit_contract.py -v
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate import _fit


class GroupsAwareModel:
    """Declares `groups`, so the harness should hand them over (and they should
    match what we passed)."""
    def fit(self, X, y, groups=None):
        self.received_groups = groups
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TabularModel:
    """Plain fit(X, y) — must be called without groups and not error."""
    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


def _data():
    X = np.zeros((6, 3))
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array([0, 0, 0, 1, 1, 1])   # two nights of 3 epochs
    return X, y, groups


def test_groups_aware_model_receives_the_subject_boundaries():
    X, y, groups = _data()
    model = _fit(GroupsAwareModel(), X, y, groups)
    assert np.array_equal(model.received_groups, groups)


def test_tabular_model_is_fit_without_groups():
    X, y, groups = _data()
    model = _fit(TabularModel(), X, y, groups)   # must not raise
    assert model.fitted is True
