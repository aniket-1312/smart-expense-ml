from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "cleaned_expenses.csv"

MODEL_FILE = BASE_DIR / "models" / "expense_model.pkl"

REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv(DATA_FILE)

df = df.dropna(
    subset=["Transaction_Text", "Label"]
)

df["Transaction_Text"] = (
    df["Transaction_Text"]
    .astype(str)
    .str.lower()
    .str.strip()
)

df["Label"] = (
    df["Label"]
    .astype(str)
    .str.strip()
)


# ==================================================
# FEATURES AND TARGET
# ==================================================

X = df["Transaction_Text"]

y = df["Label"]


# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==================================================
# LOAD VECTORIZER + MODEL
# ==================================================

vectorizer = joblib.load(
    BASE_DIR / "models" / "tfidf_vectorizer.pkl"
)

model = joblib.load(
    MODEL_FILE
)


# ==================================================
# TRANSFORM TEST DATA
# ==================================================

X_test_tfidf = vectorizer.transform(
    X_test
)


# ==================================================
# PREDICTIONS
# ==================================================

y_pred = model.predict(
    X_test_tfidf
)


# ==================================================
# METRICS
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


print("\n" + "=" * 60)

print("MODEL EVALUATION")

print("=" * 60)

print("Accuracy:", round(accuracy, 4))

print("Weighted F1:", round(weighted_f1, 4))

print("Macro F1:", round(macro_f1, 4))

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

labels = sorted(y.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)


plt.figure(figsize=(10, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Expense Category Confusion Matrix")

plt.xlabel("Predicted Category")

plt.ylabel("Actual Category")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ==================================================
# MODEL COMPARISON CHART
# ==================================================

comparison_file = BASE_DIR / "models" / "model_comparison.csv"

results_df = pd.read_csv(
    comparison_file
)

plt.figure(figsize=(12, 7))

results_df.plot(
    x="Model",
    y=["Accuracy", "Weighted_F1", "Macro_F1", "CV_Mean_F1"],
    kind="bar",
    figsize=(12, 7)
)

plt.title("ML Model Comparison")

plt.ylabel("Score")

plt.ylim(0, 1.1)

plt.xticks(rotation=15)

plt.legend(
    title="Metrics"
)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "model_comparison.png",
    dpi=300
)

plt.close()


# ==================================================
# FINAL OUTPUT
# ==================================================

print("\nReports saved successfully!")

print(
    "Confusion Matrix:",
    REPORT_DIR / "confusion_matrix.png"
)

print(
    "Model Comparison:",
    REPORT_DIR / "model_comparison.png"
)