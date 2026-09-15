from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB

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

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv(DATA_FILE)

print("\nDataset Shape:", df.shape)

print("\nDataset Columns:")
print(df.columns.tolist())


# ==================================================
# CLEAN DATA
# ==================================================


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
# DISPLAY CATEGORY DISTRIBUTION
# ==================================================

print("\nCategory Distribution:")
print(df["Label"].value_counts())


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

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==================================================
# TF-IDF VECTORIZER
# ==================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    max_features=5000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


# ==================================================
# MODELS
# ==================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        C=2.0,
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(
        alpha=0.1
    ),

    "Complement Naive Bayes": ComplementNB(
        alpha=0.1
    )

}


# ==================================================
# MODEL COMPARISON
# ==================================================

results = []

best_model = None
best_model_name = None
best_f1 = 0


for model_name, model in models.items():

    print("\n" + "=" * 60)

    print("MODEL:", model_name)

    print("=" * 60)

    # Train
    model.fit(
        X_train_tfidf,
        y_train
    )

    # Predict
    y_pred = model.predict(
        X_test_tfidf
    )

    # Metrics
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

    print("\nAccuracy:", round(accuracy, 4))

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

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    # Cross-validation
    cv_scores = cross_val_score(
        model,
        X_train_tfidf,
        y_train,
        cv=5,
        scoring="f1_weighted"
    )

    print("\n5-Fold CV F1 Scores:")

    print(cv_scores)

    print(
        "Mean CV F1:",
        round(cv_scores.mean(), 4)
    )

    # Save results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Weighted_F1": weighted_f1,
        "Macro_F1": macro_f1,
        "CV_Mean_F1": cv_scores.mean()
    })

    # Select best model
    if weighted_f1 > best_f1:

        best_f1 = weighted_f1

        best_model = model

        best_model_name = model_name


# ==================================================
# RESULTS TABLE
# ==================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Weighted_F1",
    ascending=False
)

print("\n" + "=" * 60)

print("MODEL COMPARISON SUMMARY")

print("=" * 60)

print(
    results_df.to_string(index=False)
)


# ==================================================
# SAVE BEST MODEL
# ==================================================

MODEL_FILE = MODEL_DIR / "expense_model.pkl"

VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"

joblib.dump(
    best_model,
    MODEL_FILE
)

joblib.dump(
    vectorizer,
    VECTORIZER_FILE
)


# ==================================================
# SAVE RESULTS
# ==================================================

RESULTS_FILE = MODEL_DIR / "model_comparison.csv"

results_df.to_csv(
    RESULTS_FILE,
    index=False
)


# ==================================================
# FINAL OUTPUT
# ==================================================

print("\n" + "=" * 60)

print("BEST MODEL:", best_model_name)

print("BEST WEIGHTED F1:", round(best_f1, 4))

print("=" * 60)

print("\nModel saved successfully!")

print("Model Path:", MODEL_FILE)

print("Vectorizer Path:", VECTORIZER_FILE)

print("Results Path:", RESULTS_FILE)