import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ------------------------------
# Constants
# ------------------------------
DATA_PATH = r"E:\code\FeatureShield_Phase1_clean\data\raw\train_features.csv"
MODEL_PATH = r"E:\code\FeatureShield_Phase1_clean\models\random_forest_ember.joblib"
NROWS = 2000  # adjust if needed
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ------------------------------
# Load CSV subset
# ------------------------------
print(f"Loading {DATA_PATH} nrows= {NROWS}")
df = pd.read_csv(DATA_PATH, nrows=NROWS)
print(f"Raw rows: {df.shape[0]} columns: {df.shape[1]}")

# ------------------------------
# Drop useless columns
# ------------------------------
drop_cols = ["Unnamed: 0", "avclass"]
df = df.drop(columns=drop_cols)
print(f"Dropped {len(drop_cols)} useless columns")

# ------------------------------
# Convert 'appeared' to numeric year
# ------------------------------
df["appeared_year"] = pd.to_datetime(df["appeared"], errors="coerce").dt.year
df = df.drop(columns=["appeared"])

# ------------------------------
# Convert string-list columns to numeric features
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
    df = df.drop(columns=[col, col + "_array"])

# ------------------------------
# Drop remaining object columns
# ------------------------------
obj_cols = df.select_dtypes(include="object").columns
df = df.drop(columns=obj_cols)

# ------------------------------
# Check for duplicates by sha256 (if still present)
# ------------------------------
if "sha256" in df.columns:
    duplicates = df.duplicated(subset=["sha256"]).sum()
    print(f"Dropped {duplicates} exact sha256 duplicates (if any).")
    df = df.drop_duplicates(subset=["sha256"])

# ------------------------------
# Split features and target
# ------------------------------
X = df.drop(columns=["label"])
y = df["label"]

# ------------------------------
# Stratified train/test split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,  # preserves label proportions in both sets
)

print(f"Train rows: {X_train.shape[0]} Test rows: {X_test.shape[0]}")
print(f"Final feature count: {X_train.shape[1]}")

# ------------------------------
# Train Random Forest
# ------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)

# ------------------------------
# Evaluate model
# ------------------------------
y_pred = rf.predict(X_test)
print("\nClassification report (test set):\n")
print(classification_report(y_test, y_pred))

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Training accuracy:", rf.score(X_train, y_train))
print("Test accuracy   :", rf.score(X_test, y_test))

# ------------------------------
# Save the trained model
# ------------------------------
joblib.dump(rf, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
