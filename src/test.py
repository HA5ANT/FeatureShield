# import pandas as pd
#
# # Path to your full CSV
# csv_path = r"E:\\code\\FeatureShield_Phase1_clean\\data\\raw\\train_features.csv"
#
# # Chunk size (adjust based on your RAM, e.g., 100k rows)
# chunksize = 100000
#
# # Initialize counters
# label_counts = {0: 0, 1: 0}
#
# # Process CSV in chunks
# for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=["label"]):
#     counts = chunk["label"].value_counts()
#     for label, count in counts.items():
#         if label in label_counts:
#             label_counts[label] += count
#         else:
#             label_counts[label] = count
#
# # Print results
# print("Label distribution in the full dataset:")
# for label, count in label_counts.items():
#     print(f"Label {label}: {count}")

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

print("XGBoost version:", xgb.__version__)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = xgb.XGBClassifier(tree_method="gpu_hist", predictor="gpu_predictor")
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))
print("✅ GPU test finished — if no errors, CUDA works!")

