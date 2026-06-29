"""Feature extraction, following the paper (Mallela & Mallett, 2024).

Turns one subject-night into a causal per-epoch feature matrix. Three features per
30 s epoch, each using only samples at or before the epoch end:

  0: heart rate, mean bpm in the epoch, EMA-smoothed, cubed, /1000
  1: motion, ActiGraph activity counts (agcounts; Neishabouri 2022): the
     accelerometer is resampled to 30 Hz, turned into per-second count magnitudes,
     summed within the epoch, squared, then EMA-smoothed
  2: time-of-night in hours

The counts come from agcounts; the EMA and gap-fill from pandas.

Two things the paper does not pin down. The EMA smoothing constants aren't stated
(0.30 here). The paper's final "normalize the EMA" is left out: a whole-recording
normalization would use the future, and scaling is the model's job anyway (trees
ignore it; a scale-sensitive model can fit a scaler on the training split).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from agcounts.extract import get_counts

from dataset import EPOCH_SEC, Record

HR_EMA_ALPHA = 0.30       # smoothing for heart-rate feature (not stated in the paper)
ACT_EMA_ALPHA = 0.30      # smoothing for activity feature (not stated in the paper)
ACCEL_FS = 30             # Hz; resampling grid for activity counts (the paper's rate)


def featurize(record: Record) -> np.ndarray:
    """Causal (n_epochs, n_features) feature matrix for one night."""
    hr_feat = (_ema(_epoch_hr_mean(record), HR_EMA_ALPHA) ** 3) / 1000.0
    activity_feat = _ema(_activity_counts(record) ** 2, ACT_EMA_ALPHA)   # summed, then squared
    time_of_night = record.epoch_time / 3600.0
    return np.column_stack([hr_feat, activity_feat, time_of_night])


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal exponential moving average (pandas; past-and-present only)."""
    return pd.Series(x).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def _epoch_hr_mean(record: Record) -> np.ndarray:
    """Mean heart rate per epoch, using only samples in (t - EPOCH_SEC, t]. Epochs
    with no sample are forward-filled from the past so callers never see NaN."""
    epochs = record.epoch_time
    n = epochs.size
    epoch_of = np.searchsorted(epochs, record.hr_time, side="left")
    keep = (epoch_of < n) & (record.hr_time > epochs[np.clip(epoch_of, 0, n - 1)] - EPOCH_SEC)
    idx = epoch_of[keep]
    total = np.bincount(idx, record.hr[keep], minlength=n)
    count = np.bincount(idx, minlength=n)
    mean = np.divide(total, count, out=np.full(n, np.nan), where=count > 0)
    return _causal_fill(mean)


def _activity_counts(record: Record) -> np.ndarray:
    """Per-epoch summed activity-count magnitude (paper / Neishabouri 2022).

    Resample the accelerometer to a fixed 30 Hz grid, compute activity counts per
    SECOND with agcounts, take the vector magnitude per second, and sum the
    magnitudes within each 30 s epoch. Causal: epoch i sums only seconds
    [30(i-1), 30 i). Epoch 0 (window before t=0) has no counts.
    """
    n = record.epoch_time.size
    grid = np.arange(0.0, record.epoch_time[-1], 1.0 / ACCEL_FS)
    if grid.size < ACCEL_FS:                                  # under a second of data
        return np.zeros(n)
    accel = np.column_stack(
        [np.interp(grid, record.motion_time, record.motion[:, k]) for k in range(3)])
    per_second = get_counts(accel, freq=ACCEL_FS, epoch=1)    # (seconds, 3)
    magnitude = np.linalg.norm(per_second, axis=1)            # vector addition

    epoch_of_second = np.arange(magnitude.size) // int(EPOCH_SEC) + 1
    return np.bincount(epoch_of_second, weights=magnitude, minlength=n + 1)[:n]


def _causal_fill(x: np.ndarray) -> np.ndarray:
    """Forward-fill gaps from the past (pandas); leading gaps take the first finite
    value, all-missing -> 0. Defines the missing-heart-rate case out of existence."""
    return pd.Series(x).ffill().bfill().fillna(0.0).to_numpy()
