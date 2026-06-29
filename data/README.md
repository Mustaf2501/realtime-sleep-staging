# data/ — Walch et al. (2019) Apple Watch sleep dataset

This directory holds the raw recordings. **It stays local and is never edited or
committed** (see the repo `.gitignore`). The data is required — `dataset.py`
raises if this folder is empty.

## Get the data

The dataset is "Sleep stage prediction with raw acceleration and
photoplethysmography heart rate data derived from a consumer wearable device"
(Walch, Huang, Forger & Goldstein, 2019) — the same data used in the paper
(31 subjects, one night each).

- PhysioNet: https://physionet.org/content/sleep-accel/1.0.0/
- Source repo: https://github.com/ojwalch/sleep_classifiers

Download and place the per-subject text files **directly in this folder** (flat,
no subdirectories):

```
data/
  46343_acceleration.txt
  46343_heartrate.txt
  46343_labeled_sleep.txt
  759667_acceleration.txt
  ...
```

`dataset.py` auto-discovers every `*_labeled_sleep.txt` and loads the matching
streams. Subject id = the filename prefix.

## Expected file formats

| File | Columns | Notes |
|------|---------|-------|
| `<id>_acceleration.txt` | `t(s)  x  y  z` | triaxial acceleration in g |
| `<id>_heartrate.txt`    | `t(s), bpm`     | comma-separated |
| `<id>_labeled_sleep.txt`| `t(s)  stage`   | 30 s epochs; PSG codes below |

PSG stage codes: `-1` unscored, `0` Wake, `1` N1, `2` N2, `3` N3, `4` (legacy
stage 4 → folded into N3), `5` REM. `dataset.py` maps these to the canonical
`Wake/N1/N2/N3/REM` set and treats REM as the positive class.

Once the files are present, confirm the loader sees them:

```bash
python dataset.py     # should report "Loaded N real subject-nights"
```
