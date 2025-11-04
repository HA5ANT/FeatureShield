## FeatureShield — Phase 1

Prototype for training and evaluating models on EMBER-style tabular features. Includes a CPU Random Forest pipeline and a GPU-accelerated XGBoost pipeline. Both perform lightweight preprocessing, stratified splitting, model evaluation, and export a persisted classifier.

---

### Repository structure

- `data/raw/train_features.csv`: Input features CSV (EMBER-style).
- `data/raw/explore_dataset.ipynb`: Optional EDA notebook.
- `src/training.py`: End-to-end training and evaluation script.
- `src/traininggpu.py`: End-to-end training and evaluation script (XGBoost on GPU).
- `src/test.py`: Utility to compute label distribution over the full dataset in chunks.
- `models/random_forest_ember.joblib`: Trained model artifact (output).
- `models/xgboost_ember.joblib`: Trained GPU XGBoost model artifact (output).
- `requirements.txt`: Python dependencies.

---

### Environment setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- Paths in the scripts are currently absolute to `E:\code\FeatureShield_Phase1_clean`. For portability, prefer relative paths like `data/raw/train_features.csv` and `models/random_forest_ember.joblib`.
- GPU notes: On Windows, the `xgboost` pip wheel includes GPU support. Ensure you have an NVIDIA GPU with a recent driver. No extra CUDA package is typically required beyond the driver for the wheel to use CUDA.

---

### Data expectations

The training script expects at minimum:
- `label`: target column (binary or multiclass integers).
- `appeared`: datetime-like string column (converted to `appeared_year`).
- Several string-encoded, pipe-delimited columns such as `histogram`, `byteentropy`, `paths`, `urls`, `registry`, `datadirectories`. These are summarized into numeric aggregates.
- Optional: `sha256` sample identifier (used to deduplicate if present).

---

### Train and evaluate (CPU, Random Forest)

```powershell
python src/training.py
```

Config in `src/training.py`:

```text
DATA_PATH    = E:\code\FeatureShield_Phase1_clean\data\raw\train_features.csv
MODEL_PATH   = E:\code\FeatureShield_Phase1_clean\models\random_forest_ember.joblib
NROWS        = 2000
TEST_SIZE    = 0.2
RANDOM_STATE = 42
```

What it does:
1. Loads up to `NROWS` rows from the CSV.
2. Drops `Unnamed: 0` and `avclass`.
3. Converts `appeared` to `appeared_year` and removes the original column.
4. Parses list-like string columns into float arrays, then creates `_sum` and `_mean` features and drops the raw strings.
5. Drops remaining `object`-typed columns so only numeric features remain.
6. Deduplicates by `sha256` if the column exists.
7. Performs stratified `train_test_split` and trains a `RandomForestClassifier(n_estimators=100)`.
8. Prints classification report, confusion matrix, and train/test accuracy.
9. Saves the model to `models/random_forest_ember.joblib`.

Tip: For stronger duplicate isolation, consider a group-aware split such as `GroupShuffleSplit` using `sha256` as groups.

---

### Train and evaluate (GPU, XGBoost)

```powershell
python src/traininggpu.py
```

Config in `src/traininggpu.py`:

```text
DATA_PATH    = E:\code\FeatureShield_Phase1_clean\data\raw\train_features.csv
MODEL_PATH   = E:\code\FeatureShield_Phase1_clean\models\xgboost_ember.joblib
NROWS        = 500000
TEST_SIZE    = 0.2
RANDOM_STATE = 42
```

What it does (differences vs CPU version):
- Uses `xgboost.XGBClassifier` with `tree_method="gpu_hist"` and `predictor="gpu_predictor"` for CUDA acceleration.
- Casts features to `float32` to reduce VRAM usage and match GPU precision.
- Adds optional early stopping with `eval_set=[(X_test, y_test)]` and `early_stopping_rounds=50`.
- Saves the model to `models/xgboost_ember.joblib`.

Notes:
- Keep `NROWS` within your GPU VRAM capacity. If you hit OOM, reduce `NROWS` or tune model parameters (`max_depth`, `n_estimators`).
- The preprocessing pipeline mirrors the CPU script so both models are trained on the same feature space.

---

### Inspect label distribution (full CSV)

```powershell
python src/test.py
```

`src/test.py` streams the dataset in chunks and prints counts per label, which is useful when the full file does not fit in memory.

---

### Troubleshooting

- Accuracy unusually high: verify stratification and deduplication; consider group-aware split by `sha256`.
- Memory pressure: reduce `NROWS` in `training.py` or use `src/test.py` for chunked inspection.
- Path errors: change absolute constants to relative paths in the scripts.
- GPU not detected: ensure a recent NVIDIA driver is installed. Verify GPU mode by checking that `tree_method="gpu_hist"` is set and watch for GPU utilization with tools like `nvidia-smi`.

---

### License

For educational and defensive research purposes only. No warranties.


