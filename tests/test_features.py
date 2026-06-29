"""Tests for the feature extraction in features.py.

Each test builds a small, controlled input so the expected output is obvious by
inspection. Run with:

    uv run --extra test python -m pytest tests/ -v

They cover:
  - the feature matrix has the right shape and never contains NaN
  - heart rate is bucketed into (t-30, t] windows and gaps are forward-filled
  - the EMA smoothing matches a hand-computed result
  - the activity count for epoch i comes from the time block [30(i-1), 30i)
  - features are causal: the first k epochs don't change when the rest is removed
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import EPOCH_SEC, Record, load_records
from features import (ACT_EMA_ALPHA, HR_EMA_ALPHA, _activity_counts, _causal_fill,
                      _ema, _epoch_hr_mean, featurize)

NAN = np.nan


# --------------------------------------------------------------------------- #
# helpers to build controlled synthetic recordings
# --------------------------------------------------------------------------- #
def make_record(n_epochs=20, hr_bpm=60.0, burst_epoch=None) -> Record:
    """A flat, still night (constant HR, no motion). If burst_epoch is given, a
    1 Hz movement is injected into that epoch's 30 s block [30(e-1), 30e)."""
    epoch_time = np.arange(n_epochs) * EPOCH_SEC
    stage = np.full(n_epochs, 2)                       # all N2 (scored)

    hr_time = np.arange(0.0, n_epochs * EPOCH_SEC, 5.0)
    hr = np.full(hr_time.shape, hr_bpm)

    motion_time = np.arange(0.0, n_epochs * EPOCH_SEC, 1.0 / 30)
    motion = np.zeros((motion_time.size, 3))
    motion[:, 2] = 1.0                                 # gravity on z, no movement
    if burst_epoch is not None:
        lo, hi = EPOCH_SEC * (burst_epoch - 1), EPOCH_SEC * burst_epoch
        in_block = (motion_time >= lo) & (motion_time < hi)
        motion[in_block, 0] += 0.3 * np.sin(2 * np.pi * 1.0 * motion_time[in_block])

    return Record("synthetic", epoch_time, stage, hr_time, hr, motion_time, motion)


def truncate(r: Record, k: int) -> Record:
    """The same record observed only through the end of epoch k-1."""
    t_end = r.epoch_time[k - 1]
    hm, mm = r.hr_time <= t_end, r.motion_time <= t_end
    return Record(r.subject_id, r.epoch_time[:k], r.stage[:k],
                  r.hr_time[hm], r.hr[hm], r.motion_time[mm], r.motion[mm])


# --------------------------------------------------------------------------- #
# shape / sanity
# --------------------------------------------------------------------------- #
def test_featurize_shape_one_row_per_epoch():
    r = make_record(n_epochs=20)
    X = featurize(r)
    assert X.shape == (20, 3)            # one row per epoch, three features


def test_featurize_has_no_nans():
    X = featurize(make_record(n_epochs=20))
    assert np.isfinite(X).all()          # forward-fill removes every NaN


def test_real_data_shape_and_finite():
    """Sanity check on a real subject-night from the dataset."""
    r = load_records()[0]
    X = featurize(r)
    assert X.shape == (r.epoch_time.size, 3)
    assert np.isfinite(X).all()


# --------------------------------------------------------------------------- #
# time-of-night feature (column 2)
# --------------------------------------------------------------------------- #
def test_time_of_night_is_epoch_time_in_hours():
    r = make_record(n_epochs=20)
    X = featurize(r)
    assert np.allclose(X[:, 2], r.epoch_time / 3600.0)


# --------------------------------------------------------------------------- #
# heart-rate bucketing + forward-fill
# --------------------------------------------------------------------------- #
def test_hr_mean_buckets_into_window_and_forward_fills():
    # 4 epochs ending at 0, 30, 60, 90 s. Place samples in known windows:
    #   epoch 0  (-30, 0] : sample at -5s  -> 50
    #   epoch 1  (0, 30]  : samples 10s,20s -> mean(60, 80) = 70
    #   epoch 2  (30, 60] : no samples       -> forward-fill -> 70
    #   epoch 3  (60, 90] : no samples       -> forward-fill -> 70
    #   sample at 100s is beyond the last epoch -> ignored
    r = Record(
        "hr", epoch_time=np.array([0.0, 30, 60, 90]), stage=np.array([2, 2, 2, 2]),
        hr_time=np.array([-5.0, 10, 20, 100]), hr=np.array([50.0, 60, 80, 999]),
        motion_time=np.zeros(1), motion=np.zeros((1, 3)))
    assert np.array_equal(_epoch_hr_mean(r), np.array([50.0, 70.0, 70.0, 70.0]))


# --------------------------------------------------------------------------- #
# EMA smoothing
# --------------------------------------------------------------------------- #
def test_ema_matches_hand_computation():
    # causal EMA, alpha=0.5, adjust=False:
    #   y0=0, y1=0, y2=0.5*8=4, y3=0.5*4=2, y4=0.5*2=1
    out = _ema(np.array([0.0, 0, 8, 0, 0]), alpha=0.5)
    assert np.allclose(out, [0.0, 0.0, 4.0, 2.0, 1.0])


def test_constant_hr_gives_constant_hr_feature():
    # HR steady at 60 bpm -> EMA stays 60 -> feature = 60**3 / 1000 = 216
    X = featurize(make_record(n_epochs=20, hr_bpm=60.0))
    assert np.allclose(X[:, 0], 60.0 ** 3 / 1000.0)


# --------------------------------------------------------------------------- #
# activity counts: which time block, and zero when still
# --------------------------------------------------------------------------- #
def test_activity_is_zero_with_no_movement():
    counts = _activity_counts(make_record(n_epochs=20))
    assert np.all(counts == 0.0)


def test_activity_count_localizes_to_its_epoch_block():
    # Movement injected only into block [150, 180) s = epoch 6's window.
    counts = _activity_counts(make_record(n_epochs=20, burst_epoch=6))
    assert counts[6] == counts.max()        # the spike lands on epoch 6
    assert counts[6] > 0
    assert np.all(counts[:4] == 0.0)         # quiet epochs before stay zero
    assert counts[0] == 0.0                  # epoch 0 has no preceding block


def test_featurize_activity_column_is_smoothed_squared_counts():
    # paper: the summed count magnitude is squared, then EMA-smoothed
    r = make_record(n_epochs=20, burst_epoch=6)
    X = featurize(r)
    assert np.allclose(X[:, 1], _ema(_activity_counts(r) ** 2, ACT_EMA_ALPHA))


# --------------------------------------------------------------------------- #
# forward-fill  (applied to the per-epoch HR mean during feature extraction,
# NOT to the raw record; only missing heart-rate epochs are filled)
# --------------------------------------------------------------------------- #
def test_forward_fill_carries_last_value_over_a_gap():
    raw =      np.array([60, NAN, NAN, 72, NAN, 68], float)
    expected = np.array([60, 60,  60,  72, 72,  68], float)
    assert np.array_equal(_causal_fill(raw), expected)


def test_forward_fill_leading_gap_uses_first_real_value():
    # nothing exists before the first sample, so back-fill with the first real one
    raw =      np.array([NAN, NAN, 50, 60], float)
    expected = np.array([50,  50,  50, 60], float)
    assert np.array_equal(_causal_fill(raw), expected)


def test_forward_fill_trailing_gap_holds_last_value():
    raw =      np.array([60, 70, NAN, NAN], float)
    expected = np.array([60, 70, 70,  70], float)
    assert np.array_equal(_causal_fill(raw), expected)


def test_forward_fill_leaves_complete_data_unchanged():
    raw = np.array([60, 61, 62, 63], float)
    assert np.array_equal(_causal_fill(raw), raw)


def test_forward_fill_all_missing_defaults_to_zero():
    # documented fallback: if there is no value to carry, fill with 0.0
    assert np.array_equal(_causal_fill(np.array([NAN, NAN], float)), np.array([0.0, 0.0]))


def test_forward_fill_is_only_used_on_hr_not_the_raw_record():
    # The raw record is untouched; only the per-epoch HR *mean* gets filled.
    # epoch 1's window (0, 30] has a sample; epoch 2's window (30, 60] has none,
    # so the epoch-2 mean is forward-filled from epoch 1 -- the record stays as-is.
    r = Record(
        "ff", epoch_time=np.array([0.0, 30, 60]), stage=np.array([2, 2, 2]),
        hr_time=np.array([15.0]), hr=np.array([66.0]),
        motion_time=np.zeros(1), motion=np.zeros((1, 3)))
    before = r.hr.copy()
    means = _epoch_hr_mean(r)
    assert np.array_equal(means, np.array([66.0, 66.0, 66.0]))  # filled both ways
    assert np.array_equal(r.hr, before)                          # record unchanged


# --------------------------------------------------------------------------- #
# causality: the past must not depend on the future
# --------------------------------------------------------------------------- #
def test_features_are_causal():
    # Movement at epoch 6; truncate at epoch 15 (a quiet region). The features for
    # the first 15 epochs must be identical whether or not the rest of the night
    # exists -- i.e. nothing looks ahead.
    r = make_record(n_epochs=20, burst_epoch=6)
    k = 15
    assert np.allclose(featurize(r)[:k], featurize(truncate(r, k))[:k])
