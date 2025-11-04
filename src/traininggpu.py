import pandas as pd  # DataFrame operations and CSV loading
import numpy as np  # Numerical operations on arrays and lists
import xgboost as xgb  # XGBoost classifier with GPU support
from sklearn.model_selection import train_test_split  # Utility to split data into train/test sets
from sklearn.metrics import classification_report, confusion_matrix  # Evaluation metrics
import joblib  # Model persistence (save/load trained models)

# ------------------------------
# Configuration constants
# ------------------------------
# Absolute path to the training CSV containing features and labels
DATA_PATH = r"E:\code\FeatureShield_Phase1_clean\data\raw\train_features.csv"
# Absolute path where the trained XGBoost model will be saved (native XGBoost format)
MODEL_PATH = r"E:\code\FeatureShield_Phase1_clean\models\xgboost_ember.json"
# Number of rows to read from the CSV for faster experimentation (use None to read all)
NROWS = 500000  # Adjust upward for better performance estimates if compute allows
# Proportion of samples to include in the test split (20% test, 80% train)
TEST_SIZE = 0.2
# Seed for reproducibility across runs (affects splitting and model initialization)
RANDOM_STATE = 42

# ------------------------------
# Load a subset (or full set) of rows from the CSV into a DataFrame
# ------------------------------
print(f"Loading {DATA_PATH} nrows= {NROWS}")
df = pd.read_csv(DATA_PATH, nrows=NROWS)
print(f"Raw rows: {df.shape[0]} columns: {df.shape[1]}")

# ------------------------------
# Drop columns known to be non-informative for modeling
# ------------------------------
drop_cols = ["Unnamed: 0", "avclass"]
df = df.drop(columns=drop_cols)
print(f"Dropped {len(drop_cols)} useless columns")

# ------------------------------
# Convert the timestamp-like 'appeared' column into a numeric year feature
# ------------------------------
df["appeared_year"] = pd.to_datetime(df["appeared"], errors="coerce").dt.year
df = df.drop(columns=["appeared"])  # remove the original textual datetime column

# ------------------------------
# Convert string-encoded lists (pipe-delimited) into aggregate numeric features
# ------------------------------
list_cols = ["histogram", "byteentropy", "paths", "urls", "registry", "datadirectories"]
for col in list_cols:
    df[col + "_array"] = df[col].apply(
        lambda x: [
            float(i) if i.replace(".", "", 1).isdigit() else 0
            for i in str(x).split("|")
        ]
    )
    df[col + "_sum"] = df[col + "_array"].apply(np.sum)
    df[col + "_mean"] = df[col + "_array"].apply(np.mean)
    df = df.drop(columns=[col, col + "_array"])  # drop original and temp array columns

# ------------------------------
# Drop any remaining non-numeric (object) columns to ensure the model sees only numbers
# ------------------------------
obj_cols = df.select_dtypes(include="object").columns
df = df.drop(columns=obj_cols)

# ------------------------------
# Remove exact duplicate samples identified by the 'sha256' identifier, if available
# ------------------------------
if "sha256" in df.columns:
    duplicates = df.duplicated(subset=["sha256"]).sum()
    print(f"Dropped {duplicates} exact sha256 duplicates (if any).")
    df = df.drop_duplicates(subset=["sha256"])

# ------------------------------
# Separate features matrix X and target vector y
# ------------------------------
X = df.drop(columns=["label"])
y = df["label"]

# Clean numeric features: replace inf/-inf with NaN, then fill NaNs
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# XGBoost/GPU prefers float32 inputs to save memory and match GPU precision
X = X.astype(np.float32)

# ------------------------------
# Create stratified train/test splits to preserve class distribution
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print(f"Train rows: {X_train.shape[0]} Test rows: {X_test.shape[0]}")
print(f"Final feature count: {X_train.shape[1]}")

# ------------------------------
# Handle class imbalance (binary only) via scale_pos_weight = neg/pos
# ------------------------------
unique_labels = np.unique(y_train)
scale_pos_weight = None
if unique_labels.shape[0] == 2:
    num_pos = (y_train == unique_labels.max()).sum()
    num_neg = (y_train == unique_labels.min()).sum()
    if num_pos > 0:
        scale_pos_weight = float(num_neg) / float(num_pos)
        print(f"Computed scale_pos_weight (binary): {scale_pos_weight:.4f}")

# ------------------------------
# Initialize and train an XGBoost classifier on the GPU
# ------------------------------
gpu_params = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    tree_method="gpu_hist",     # GPU training
    predictor="gpu_predictor",  # GPU inference
    eval_metric="logloss",
    n_jobs=0,                    # let the GPU do the work
)
if scale_pos_weight is not None:
    gpu_params["scale_pos_weight"] = scale_pos_weight

cpu_params = dict(gpu_params)
cpu_params["tree_method"] = "hist"
cpu_params["predictor"] = "auto"

model = xgb.XGBClassifier(**gpu_params)

# Optional: early stopping for faster/better training; disable verbose printing
try:
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=50,
    )
except Exception as e:
    print(f"GPU training failed ({e}). Falling back to CPU 'hist' tree method.")
    model = xgb.XGBClassifier(**cpu_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
        early_stopping_rounds=50,
    )

# ------------------------------
# Evaluate the trained model on the held-out test set
# ------------------------------
y_pred = model.predict(X_test)
print("\nClassification report (test set):\n")
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Training accuracy:", model.score(X_train, y_train))
print("Test accuracy   :", model.score(X_test, y_test))

# ------------------------------
# Persist the trained model to disk for later inference/use (native XGBoost format)
# ------------------------------
model.save_model(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")


