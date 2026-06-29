# Real-time sleep staging

A harness for optimizing a real-time REM detector with [Weco](https://www.weco.ai/),
using the Walch et al. (2019) Apple Watch dataset (heart rate, motion, and
time-of-night, to REM vs. not-REM), following Mallela & Mallett (2024).

Weco rewrites `module.py` to raise the REM F1 under leave-one-subject-out
cross-validation. Each candidate model has to stay causal, so it can run live on
the watch rather than reading the whole night after the fact.

## Layout

| File | Role | Weco edits? |
|------|------|-------------|
| `module.py`   | the model: classifier plus REM threshold | yes |
| `features.py` | feature extraction (heart rate, activity counts, time) | no |
| `splits.py`   | builds the feature matrix and the LOSO splitter | no |
| `evaluate.py` | scores via LOSO, prints `metric: N`, saves results | no |
| `dataset.py`  | loads the Walch recordings (parsed once, cached) | no |
| `data/`       | recordings and the committed feature matrix (see `data/README.md`) | no |
| `results/`    | per-model confusion matrix and metrics | no |

`module.py` must provide `build_model()`, returning a scikit-learn estimator with
`fit(X, y)` and `predict(X)` (1 for REM). The features are in `features.py` and
Weco does not change them; it optimizes only the model. `evaluate.py` checks each
fold for look-ahead and scores 0 if a model's earlier predictions change when
later epochs are removed.

### Design

The modules keep simple interfaces over the detail: `dataset` handles file parsing
and caching, `features` the feature extraction, `module` the model, and `splits`
the dataset and folds. Splitting and scoring use `sklearn.model_selection` and
`sklearn.metrics` rather than hand-written code.

### Models Weco can try

Any scikit-learn-compatible classifier over the feature matrix: random forest,
gradient boosting (XGBoost or LightGBM), SVM, an MLP, or a torch/keras model
wrapped as an estimator. A model that needs temporal context can declare
`fit(X, y, groups)` and process each night causally. The environment already
includes scikit-learn, PyTorch, XGBoost, LightGBM, and TensorFlow/Keras. Deep
models over the raw signals are out of scope while features are fixed; edit
`features.py` to change that.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The repo ships `data/featurematrix.npz`, so the search runs without the raw
recordings. To rebuild the features from scratch, install the dataset (see
`data/README.md`) and run `evaluate.py`; a change to `features.py` triggers a
rebuild.

## Baseline

```bash
uv run python evaluate.py
```

Prints the metric and the per-subject accuracy, precision, recall, and F1, and
writes `results/<model-hash>.{py,json,png}`.

## Run Weco

```bash
weco run \
  --source module.py \
  --eval-command "uv run python evaluate.py" \
  --metric metric \
  --goal maximize \
  --steps 20 \
  --save-logs
```
