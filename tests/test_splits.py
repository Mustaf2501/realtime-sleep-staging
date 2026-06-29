"""Tests for the evaluation protocol in splits.py.

Each test builds a couple of tiny subject-nights with a known stage layout, so the
expected (X, y, groups) and the leave-one-subject-out folds can be checked by hand.

    uv run --extra test python -m pytest tests/test_splits.py -v

They cover:
  - make_dataset stacks one row per *scored* epoch (unscored -1 epochs dropped)
  - y is 1 exactly on REM epochs, groups carry the subject index
  - X rows are the fixed features of the scored epochs
  - the cross-validator is true leave-one-subject-out: one fold per subject,
    train/test never share a subject, every epoch is tested exactly once
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import features
import splits
from dataset import EPOCH_SEC, N2, N3, REM, WAKE, Record

UNSCORED = -1


def make_record(subject_id: str, stages: list[int]) -> Record:
    """A still, constant-HR night whose epochs carry the given stage codes."""
    n = len(stages)
    epoch_time = np.arange(n) * EPOCH_SEC
    hr_time = np.arange(0.0, n * EPOCH_SEC, 5.0)
    hr = np.full(hr_time.shape, 60.0)
    motion_time = np.arange(0.0, n * EPOCH_SEC, 1.0 / 30)
    motion = np.zeros((motion_time.size, 3))
    motion[:, 2] = 1.0
    return Record(subject_id, epoch_time, np.array(stages), hr_time, hr,
                  motion_time, motion)


# --------------------------------------------------------------------------- #
# make_dataset
# --------------------------------------------------------------------------- #
def test_make_dataset_stacks_labels_and_groups():
    # s0: [Wake, REM, unscored, REM]  -> scored epochs 0,1,3 -> y = [0, 1, 1]
    # s1: [N2,   REM, Wake]           -> all scored          -> y = [0, 1, 0]
    records = [make_record("s0", [WAKE, REM, UNSCORED, REM]),
               make_record("s1", [N2, REM, WAKE])]
    X, y, groups = splits.make_dataset(records)

    assert X.shape == (6, 3)                                   # 6 scored epochs, 3 features
    assert np.array_equal(y, [0, 1, 1, 0, 1, 0])              # 1 exactly on REM
    assert np.array_equal(groups, [0, 0, 0, 1, 1, 1])        # subject index per row


def test_make_dataset_drops_unscored_epochs():
    # 5 epochs, 2 of them unscored -> only 3 rows survive
    records = [make_record("s", [WAKE, UNSCORED, REM, UNSCORED, N2])]
    X, y, groups = splits.make_dataset(records)
    assert X.shape[0] == 3
    assert np.array_equal(y, [0, 1, 0])                       # Wake, REM, N2


def test_make_dataset_rows_are_features_of_scored_epochs():
    # X for a subject must equal features.featurize(record) restricted to scored rows
    r = make_record("s", [WAKE, REM, UNSCORED, N3, REM])
    X, _, _ = splits.make_dataset([r])
    expected = features.featurize(r)[r.scored_mask]
    assert np.allclose(X, expected)


def test_make_dataset_aligned_lengths():
    records = [make_record("s0", [REM, WAKE, N2]),
               make_record("s1", [N2, REM, WAKE, N3])]
    X, y, groups = splits.make_dataset(records)
    assert len(X) == len(y) == len(groups) == 7              # 3 + 4 scored epochs


def test_load_dataset_uses_saved_matrix_without_raw_data(tmp_path, monkeypatch):
    # Point the committed-matrix path at a temp file and save a synthetic matrix.
    monkeypatch.setattr(splits, "DATASET_FILE", str(tmp_path / "featurematrix.npz"))
    X = np.arange(12, dtype=float).reshape(6, 2)
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1, 2, 2])
    splits._save(X, y, groups, splits._features_hash())

    # If load_dataset touched the raw recordings, this would fire — it must not.
    import dataset
    def _boom():
        raise AssertionError("load_dataset read raw data instead of the saved matrix")
    monkeypatch.setattr(dataset, "load_records", _boom)

    Xo, yo, go = splits.load_dataset()
    assert np.array_equal(Xo, X) and np.array_equal(yo, y) and np.array_equal(go, groups)


# --------------------------------------------------------------------------- #
# leave-one-subject-out cross-validator
# --------------------------------------------------------------------------- #
# Three subjects of different sizes so folds are easy to tell apart.
THREE = [make_record("s0", [REM, WAKE, N2]),               # 3 scored
         make_record("s1", [N2, REM, WAKE, N3]),           # 4 scored
         make_record("s2", [WAKE, N2, REM, N3, REM])]      # 5 scored


def test_loso_one_fold_per_subject():
    X, y, groups = splits.make_dataset(THREE)
    assert splits.cross_validator().get_n_splits(groups=groups) == 3


def test_loso_train_and_test_never_share_a_subject():
    X, y, groups = splits.make_dataset(THREE)
    held_out = []
    for train_idx, test_idx in splits.cross_validator().split(X, y, groups=groups):
        test_subjects = set(groups[test_idx])
        train_subjects = set(groups[train_idx])
        assert len(test_subjects) == 1                       # test = exactly one subject
        assert test_subjects.isdisjoint(train_subjects)      # never in its own training set
        held_out.append(test_subjects.pop())
    assert sorted(held_out) == [0, 1, 2]                     # each subject held out once


def test_loso_each_epoch_is_tested_exactly_once():
    X, y, groups = splits.make_dataset(THREE)
    tested = np.concatenate([te for _, te in splits.cross_validator().split(X, y, groups=groups)])
    assert sorted(tested) == list(range(len(X)))             # full coverage, no overlap


def test_loso_test_fold_sizes_match_subject_sizes():
    X, y, groups = splits.make_dataset(THREE)
    sizes = sorted(len(te) for _, te in splits.cross_validator().split(X, y, groups=groups))
    assert sizes == [3, 4, 5]                                # the three subjects' epoch counts
