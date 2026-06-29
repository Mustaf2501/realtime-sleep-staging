"""Tests that the dataset's epochs are in chronological order.

Sequence models and the causality guard assume each night runs earliest to latest.
The loader never re-sorts, so these check the data actually holds that order.

    uv run --extra test python -m pytest tests/test_dataset.py -v
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import EPOCH_SEC, load_records


def test_each_night_is_in_chronological_order():
    for r in load_records():
        diffs = np.diff(r.epoch_time)
        assert np.all(diffs > 0), f"{r.subject_id}: epoch_time is not strictly increasing"


def test_epochs_lie_on_a_regular_30s_grid():
    for r in load_records():
        diffs = np.diff(r.epoch_time)
        assert np.allclose(diffs, EPOCH_SEC), f"{r.subject_id}: epochs are not 30 s apart"
