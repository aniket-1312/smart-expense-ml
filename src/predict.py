import joblib

# Load trained model
model = joblib.load("models/expense_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


def predict_expense(text):
    # Convert text into TF-IDF features
    text_tfidf = vectorizer.transform([text])

    # Predict category
    prediction = model.predict(text_tfidf)

    # Get prediction probability
    probabilities = model.predict_proba(text_tfidf)

    confidence = probabilities.max()

    return prediction[0], confidence

# ==================================================
# SMART CATEGORY RULES
# ==================================================

CATEGORY_KEYWORDS = {
    "Food": [
        "swiggy",
        "zomato",
        "restaurant",
        "dinner",
        "lunch",
        "breakfast",
        "food",
        "grocery",
        "pizza",
        "dominos",
        "mcdonald",
    ],

    "Travel": [
        "uber",
        "ola",
        "cab",
        "taxi",
        "flight",
        "train ticket",
        "bus ticket",
        "metro",
        "petrol",
        "fuel",
        "parking",
    ],

    "Shopping": [
        "amazon",
        "flipkart",
        "myntra",
        "shopping",
        "clothes",
        "shoes",
        "electronics",
        "mall",
        "purchase",
    ],

    "EMI": [
        "emi",
        "loan",
        "installment",
        "monthly payment",
        "electricity bill",
        "credit card bill",
    ],

    "Investment": [
        "sip",
        "mutual fund",
        "investment",
        "stocks",
        "shares",
        "nps",
        "fd",
        "fixed deposit",
    ],
}


def keyword_category(text):
    """
    Return category if a known keyword is found.
    Otherwise return None.
    """

    text = text.lower().strip()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return category

    return None

def smart_predict(expense_text, model, vectorizer):
    """
    Hybrid prediction:
    1. Check known keywords
    2. Otherwise use ML model
    """

    # --------------------------------------------------
    # STEP 1: KEYWORD CHECK
    # --------------------------------------------------

    rule_category = keyword_category(expense_text)

    if rule_category is not None:
        return rule_category, 1.0, "Keyword Rule"

    # --------------------------------------------------
    # STEP 2: ML MODEL
    # --------------------------------------------------

    text_vector = vectorizer.transform(
        [expense_text]
    )

    predicted_category = model.predict(
        text_vector
    )[0]

    # --------------------------------------------------
    # STEP 3: CONFIDENCE
    # --------------------------------------------------

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            text_vector
        )[0]

        confidence = max(probabilities)

    else:
        confidence = 0.0

    return (
        predicted_category,
        confidence,
        "ML Model"
    )

# ==================================================
# TEST PREDICTIONS
# ==================================================

test_expenses = [
    "Swiggy dinner",
    "Uber cab payment",
    "Amazon shopping",
    "Monthly loan installment",
    "SIP investment",
    "Flipkart shopping",
    "Zomato food order",
    "Petrol for car",
    "Electricity bill payment",
    "Movie ticket",
]

for expense in test_expenses:

    category, confidence, source = smart_predict(
        expense,
        model,
        vectorizer
    )

    print("-" * 50)

    print("Expense:", expense)

    print("Predicted Category:", category)

    print("Confidence:", f"{confidence:.2%}")

    print("Prediction Source:", source)