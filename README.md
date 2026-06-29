# Real-time sleep staging — Weco optimization harness

Optimize a **real-time REM-detection** architecture for targeted lucidity
reactivation, using the Walch et al. (2019) Apple Watch dataset (heart rate +
motion + time-of-night → REM vs. not-REM), as in Mallela & Mallett (2024).

Weco rewrites `module.py` to maximize **REM F1** under **leave-one-subject-out**
cross-validation, while every change stays **causal** (deployable in the live app).

## Layout

| File | Role | Weco edits? |
|------|------|-------------|
| `module.py`  | **the model only** — classifier + REM threshold | **yes** |
| `features.py`| fixed, trusted feature extraction (HR, activity counts, time) | no |
| `evaluate.py`| scores via leave-one-subject-out CV, prints `metric: N` | no |
| `splits.py`  | builds the fixed feature matrix + LOSO cross-validator | no |
| `dataset.py` | loads the Walch recordings (parsed once, cached as `.npz`) | no |
| `data/`      | raw recordings — local only, never edited (see `data/README.md`) | no |
| `.runs/`     | Weco logs, created when you pass `--save-logs` | no |

The contract `module.py` must keep: `build_model() -> a scikit-learn-compatible
estimator` (`fit(X, y)` on feature rows, `predict(X) -> 1` for REM). Features are
**fixed** in `features.py` and out of Weco's reach; Weco optimizes only the model
(any sklearn-compatible classifier — RF, gradient boosting, SVM, MLP, or an
sklearn-wrapped torch/keras model). Because the features are causal and the model
scores each epoch independently, results stay real-time-honest; `evaluate.py`
**enforces** this by checking the model's predictions for early epochs don't
change when later epochs are removed, and scores 0 otherwise.

### Design

The code follows *A Philosophy of Software Design* (Ousterhout): a few **deep
modules** with simple interfaces hiding the complexity. `dataset` hides file
parsing and caching; `features` hides the (fixed) feature engineering; `module`
exposes just the model; `splits` owns the leave-one-subject-out dataset + folds.
Splitting and scoring are delegated to trusted libraries
(`sklearn.model_selection`, `sklearn.metrics`) rather than hand-rolled.

### Exploring models

Weco optimizes the classifier in `module.py` over the fixed feature matrix — a
random forest, gradient boosting (XGBoost/LightGBM), SVM, an MLP, or an
sklearn-wrapped torch/keras model. The base env already includes scikit-learn,
PyTorch, XGBoost, LightGBM, and TensorFlow/Keras; keep `predict(X)` per-epoch and
deterministic so the causality check stays green. Feature extraction is fixed in
`features.py` (deep raw-signal models are out of scope by design — change
`features.py` yourself if you want to revisit that).

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Add the dataset to `data/` (see `data/README.md`) — it is required; `evaluate.py`
raises if `data/` is empty.

## Establish the baseline

```bash
uv run python evaluate.py
```

This runs leave-one-subject-out CV and prints `metric: <REM F1>` plus a
per-class breakdown (recall, precision). Commit `module.py` before each
optimization run.

## Optimize with Weco

```bash
weco run \
  --source module.py \
  --eval-command "uv run python evaluate.py" \
  --metric metric \
  --goal maximize \
  --steps 20 \
  --save-logs
```
