"""Leave-one-subject-out evaluation protocol.

Sleep epochs from one subject are highly correlated, so an honest score requires
keeping each subject entirely in train OR test. The paper does exactly this with
leave-one-participant-out cross-validation, and so do we — using scikit-learn's
LeaveOneGroupOut (subject = group). It is deterministic: every subject is the
test set exactly once, so there is no seed.

THIS FILE IS NOT OPTIMIZED BY WECO. It fixes the protocol so scores stay
comparable across optimization steps, and is the single source of truth for the
evaluation dataset: feature rows, labels, and groups are built together so they
can never drift out of alignment. Features come from the fixed features.py.

The feature matrix is built once and saved to data/featurematrix.npz. Because the
features are fixed, that file is identical on every Weco step — and small enough
to COMMIT. Ship it with the repo and the search runs on another machine WITHOUT
the 1.7 GB of raw recordings: the model only ever needs (X, y, groups). The file
stores a hash of features.py; if features.py changes, the matrix is rebuilt from
./data (which must then be present).
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
