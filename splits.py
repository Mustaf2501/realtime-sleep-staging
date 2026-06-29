"""The leave-one-subject-out evaluation dataset and splitter.

Epochs from one subject are highly correlated, so a subject is kept entirely in
train or test. This is the paper's leave-one-participant-out scheme, using
scikit-learn's LeaveOneGroupOut with subject as the group. It is deterministic:
each subject is held out once, so there is no seed.

make_dataset builds (X, y, groups) together so the rows stay aligned; the features
come from features.py.

The feature matrix is saved to data/featurematrix.npz. It is the same on every
Weco step (the features are fixed) and small enough to commit, so the search can
run on another machine from the file alone, without the raw recordings. The file
stores a hash of features.py; if features.py changes it is rebuilt from ./data.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

import features
from dataset import DATA_DIR, REM, Record

DATASET_FILE = os.path.join(DATA_DIR, "featurematrix.npz")   # small; committed to git


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, groups). Use the saved feature matrix when it matches the
    current features.py — no raw recordings needed — else build it from ./data."""
    current = _features_hash()
    if os.path.exists(DATASET_FILE):
        z = np.load(DATASET_FILE)
        if str(z["features_hash"]) == current:
            return z["X"], z["y"], z["groups"]
        print("[splits] features.py changed since data/featurematrix.npz was built; "
              "rebuilding from ./data (raw recordings required).")

    from dataset import load_records          # only needed when (re)building
    X, y, groups = make_dataset(load_records())
    _save(X, y, groups, current)
    return X, y, groups


def make_dataset(records: list[Record]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack all scored epochs into (X, y, groups) for grouped cross-validation.

    X      : (n_epochs, n_features) fixed causal features
    y      : (n_epochs,) int labels, 1 == REM
    groups : (n_epochs,) subject index, so LeaveOneGroupOut holds one out

    Pure (always recomputes); load_dataset() is the cached/committable entry point.
    """
    X_parts, y_parts, group_parts = [], [], []
    for subject_index, record in enumerate(records):
        scored = record.scored_mask
        X_parts.append(features.featurize(record)[scored])
        y_parts.append((record.stage[scored] == REM).astype(int))
        group_parts.append(np.full(int(scored.sum()), subject_index))
    return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(group_parts)


def cross_validator() -> LeaveOneGroupOut:
    """The leave-one-subject-out splitter used for every evaluation."""
    return LeaveOneGroupOut()


def _features_hash() -> str:
    return hashlib.sha256(open(features.__file__, "rb").read()).hexdigest()[:16]


def _save(X: np.ndarray, y: np.ndarray, groups: np.ndarray, features_hash: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = f"{DATASET_FILE}.{os.getpid()}.tmp.npz"            # atomic write
    np.savez_compressed(tmp, X=X, y=y, groups=groups,
                        features_hash=np.array(features_hash))
    os.replace(tmp, DATASET_FILE)


if __name__ == "__main__":
    X, y, groups = load_dataset()
    print(f"{X.shape[0]} epochs x {X.shape[1]} features, {len(np.unique(groups))} subjects")
    print(f"REM prevalence: {100 * y.mean():.1f}% | LOSO folds: "
          f"{cross_validator().get_n_splits(groups=groups)}")
    print(f"saved: {DATASET_FILE} ({os.path.getsize(DATASET_FILE) / 1024:.0f} KB)")
