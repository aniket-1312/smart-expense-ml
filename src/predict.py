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


# Test examples
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
    category, confidence = predict_expense(expense)

    print(f"Expense: {expense}")
    print(f"Predicted Category: {category}")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 40)