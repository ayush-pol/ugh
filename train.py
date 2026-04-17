"""
train.py  –  Heart Disease Detection
Run this once locally (or in Colab) to produce:
  • model.pkl          (trained LR pipeline)
  • label_encoders.pkl (fitted LabelEncoders for each categorical column)
  • scaler.pkl         (fitted StandardScaler)
  • test_data.csv      (20 % hold-out set, NEVER seen by the model)
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
from imblearn.over_sampling import SMOTE

# ── 1. Load data ────────────────────────────────────────────────────────────
df = pd.read_csv("heart_disease_cleaned.csv")
TARGET = "Heart Disease Status"

# ── 2. Encode categoricals ──────────────────────────────────────────────────
cat_cols = df.select_dtypes(include="object").columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ── 3. Split BEFORE SMOTE  (test set stays real-distribution) ───────────────
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Save test split (raw, unscaled) for reference
test_df = X_test.copy()
test_df[TARGET] = y_test
test_df.to_csv("test_data.csv", index=False)
print(f"Test set saved → test_data.csv  ({len(test_df)} rows)")

# ── 4. Scale ────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. SMOTE on training set only ───────────────────────────────────────────
print("\nClass distribution BEFORE SMOTE:")
print(pd.Series(y_train).value_counts().to_string())

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train_sc, y_train)

print("\nClass distribution AFTER SMOTE:")
print(pd.Series(y_res).value_counts().to_string())

# ── 6. Train Logistic Regression ────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_res, y_res)

# ── 7. Evaluate on untouched test set ───────────────────────────────────────
y_pred  = model.predict(X_test_sc)
y_proba = model.predict_proba(X_test_sc)[:, 1]

print("\n── TEST SET RESULTS ──────────────────────────────────────────────────")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Heart Disease"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 8. Persist artefacts ────────────────────────────────────────────────────
with open("model.pkl",          "wb") as f: pickle.dump(model,          f)
with open("label_encoders.pkl", "wb") as f: pickle.dump(label_encoders, f)
with open("scaler.pkl",         "wb") as f: pickle.dump(scaler,         f)

print("\n✅  model.pkl, label_encoders.pkl, scaler.pkl saved.")
