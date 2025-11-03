## FeatureShield — Phase 1

Prototype for training and evaluating a Random Forest on EMBER-style tabular features. The pipeline performs lightweight preprocessing, stratified splitting, model evaluation, and exports a persisted classifier.

---

### Repository structure

- `data/raw/train_features.csv`: Input features CSV (EMBER-style).
- `data/raw/explore_dataset.ipynb`: Optional EDA notebook.
- `src/training.py`: End-to-end training and evaluation script.
- `src/test.py`: Utility to compute label distribution over the full dataset in chunks.
- `models/random_forest_ember.joblib`: Trained model artifact (output).
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

---

### Data expectations

The training script expects at minimum:
- `label`: target column (binary or multiclass integers).
- `appeared`: datetime-like string column (converted to `appeared_year`).
- Several string-encoded, pipe-delimited columns such as `histogram`, `byteentropy`, `paths`, `urls`, `registry`, `datadirectories`. These are summarized into numeric aggregates.
- Optional: `sha256` sample identifier (used to deduplicate if present).

---

### Train and evaluate

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

---

### License

For educational and defensive research purposes only. No warranties.


