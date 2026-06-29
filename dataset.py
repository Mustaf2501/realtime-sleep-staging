"""Data loading for the Walch et al. (2019) Apple Watch sleep dataset.

"Sleep stage prediction with raw acceleration and photoplethysmography heart rate
data derived from a consumer wearable device" (Walch et al., 2019), on PhysioNet:
https://physionet.org/content/sleep-accel/1.0.0/ . See data/README.md to install.

31 subjects, one night each. Per subject, three whitespace/comma text files share
one time reference (seconds from lights-off; streams may extend to negative times
before scoring began):
    <id>_acceleration.txt   t  x  y  z      triaxial accel in g, ~30 Hz
    <id>_heartrate.txt      t, bpm          heart rate, ~0.2 Hz
    <id>_labeled_sleep.txt  t  stage        PSG hypnogram, one stage per 30 s epoch

Raw PSG stage codes (-1 unscored, 0 Wake, 1 N1, 2 N2, 3 N3, 4 legacy->N3, 5 REM)
are mapped to the canonical codes below, with REM as the positive class.

This module only loads data; the model and feature logic live elsewhere.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Canonical stage codes used everywhere downstream.
WAKE, N1, N2, N3, REM = 0, 1, 2, 3, 4
EPOCH_SEC = 30.0
_STAGE_MAP = {0: WAKE, 1: N1, 2: N2, 3: N3, 4: N3, 5: REM}  # missing -> -1 (unscored)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Raw text is parsed once and cached here as .npz; the raw files never change, so
# later loads are near-instant. This keeps every Weco evaluation fast.
CACHE_DIR = os.path.join(DATA_DIR, ".cache")


@dataclass
class Record:
    """One subject-night of raw wearable streams + PSG hypnogram.

    All times are seconds from lights-off. Streams are causal: feature code in
    module.py must only use samples with time <= the end of the current epoch.

    Nominal sampling rates (Walch et al., 2019): heart rate ~0.2 Hz (~1 sample
    per 5 s), triaxial acceleration ~30 Hz, hypnogram one stage per 30 s epoch
    (EPOCH_SEC). Actual sample timing is irregular, so feature code bins the
    streams onto the epoch grid rather than assuming a fixed rate.
    """

    subject_id: str
    epoch_time: np.ndarray   # (n_epochs,)   end-of-epoch time, every 30 s (EPOCH_SEC)
    stage: np.ndarray        # (n_epochs,)   canonical stage code, -1 = unscored
    hr_time: np.ndarray      # (n_hr,)        heart-rate sample times (~0.2 Hz)
    hr: np.ndarray           # (n_hr,)        bpm
    motion_time: np.ndarray  # (n_motion,)    accel sample times (~30 Hz)
    motion: np.ndarray       # (n_motion, 3)  triaxial acceleration in g

    @property
    def scored_mask(self) -> np.ndarray:
        return self.stage >= 0


def load_records() -> list[Record]:
    """Every subject-night in ./data, keyed by the <id>_labeled_sleep.txt files."""
    paths = glob.glob(os.path.join(DATA_DIR, "*_labeled_sleep.txt"))
    print(paths)
    ids = sorted(os.path.basename(p).replace("_labeled_sleep.txt", "") for p in paths)
    if not ids:
        raise FileNotFoundError(
            f"No recordings in {DATA_DIR}. See data/README.md to install the dataset.")
    return [_load_one(sid) for sid in ids]


def _load_one(subject_id: str) -> Record:
    """Load one subject-night, building the .npz cache on first use."""
    cache = os.path.join(CACHE_DIR, f"{subject_id}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return Record(subject_id, z["epoch_time"], z["stage"], z["hr_time"],
                      z["hr"], z["motion_time"], z["motion"])

    labels = _read(subject_id, "labeled_sleep", sep=r"\s+")  # t  stage
    hr = _read(subject_id, "heartrate", sep=",")             # t, bpm
    accel = _read(subject_id, "acceleration", sep=r"\s+")    # t  x  y  z
    record = Record(
        subject_id=subject_id,
        epoch_time=labels[:, 0],
        stage=np.array([_STAGE_MAP.get(int(c), -1) for c in labels[:, 1]]),
        hr_time=hr[:, 0], hr=hr[:, 1],
        motion_time=accel[:, 0], motion=accel[:, 1:4],
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache, epoch_time=record.epoch_time, stage=record.stage,
             hr_time=record.hr_time, hr=record.hr,
             motion_time=record.motion_time, motion=record.motion)
    return record


def _read(subject_id: str, suffix: str, sep: str) -> np.ndarray:
    """Parse one raw text file into a 2-D float array."""
    path = os.path.join(DATA_DIR, f"{subject_id}_{suffix}.txt")
    return pd.read_csv(path, header=None, sep=sep, engine="c").to_numpy(float)


if __name__ == "__main__":
    recs = load_records()
    n = sum(int(r.scored_mask.sum()) for r in recs)
    rem = sum(int((r.stage == REM).sum()) for r in recs)
    print(f"{len(recs)} subjects, {n} scored epochs, "
          f"{rem} REM epochs ({100 * rem / n:.1f}%)")
