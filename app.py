import streamlit as st
import pandas as pd
import joblib
import os

from datetime import date
import plotly.express as px


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Expense Tracker",
    page_icon="💰",
    layout="wide"
)


# --------------------------------------------------
# Load ML Model
# --------------------------------------------------

model = joblib.load("models/expense_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# --------------------------------------------------
# File Configuration
# --------------------------------------------------

EXPENSE_FILE = "data/user_expenses.csv"

# Create data directory if it does not exist
os.makedirs("data", exist_ok=True)


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def load_expenses():

    if os.path.exists(EXPENSE_FILE):

        df = pd.read_csv(EXPENSE_FILE)

        if not df.empty:
            df["Date"] = pd.to_datetime(df["Date"])

        return df

    return pd.DataFrame(
        columns=["Date", "Description", "Amount", "Category"]
    )

def save_expense(expense_date, description, amount, category):

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    new_expense = pd.DataFrame({
        "Date": [expense_date],
        "Description": [description],
        "Amount": [amount],
        "Category": [category]
    })

    if os.path.exists(EXPENSE_FILE):

        new_expense.to_csv(
            EXPENSE_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        new_expense.to_csv(
            EXPENSE_FILE,
            index=False
        )

def predict_category(description):

    text_tfidf = vectorizer.transform([description])

    prediction = model.predict(text_tfidf)[0]

    probabilities = model.predict_proba(text_tfidf)

    confidence = probabilities.max()

    return prediction, confidence


def delete_expense(index):

    df = load_expenses()

    df = df.drop(index)

    df.to_csv(
        EXPENSE_FILE,
        index=False
    )


# --------------------------------------------------
# App Header
# --------------------------------------------------

st.title("💰 Smart Expense Tracker")

st.write(
    "Track expenses, automatically categorize them using Machine Learning, "
    "and analyze your spending."
)

st.divider()


# --------------------------------------------------
# Add Expense Section
# --------------------------------------------------

st.subheader("➕ Add New Expense")

col1, col2, col3 = st.columns(3)

with col1:

    expense_date = st.date_input(
        "Expense Date",
        value=date.today()
    )

with col2:

    description = st.text_input(
        "Expense Description",
        placeholder="Example: Swiggy dinner"
    )

with col3:

    amount = st.number_input(
        "Amount (₹)",
        min_value=0.0,
        step=10.0,
        format="%.2f"
    )


if st.button(
    "🔍 Predict & Save Expense",
    use_container_width=True
):

    if description.strip() == "":

        st.warning("Please enter an expense description.")

    elif amount <= 0:

        st.warning("Please enter a valid amount.")

    else:

        category, confidence = predict_category(description)

        save_expense(
            expense_date,
            description,
            amount,
            category
        )

        st.success("Expense saved successfully!")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Category",
                category
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )


# --------------------------------------------------
# Load Expense Data
# --------------------------------------------------

expenses = load_expenses()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("⚙️ Settings")

monthly_budget = st.sidebar.number_input(
    "Monthly Budget (₹)",
    min_value=0.0,
    value=10000.0,
    step=500.0
)


# --------------------------------------------------
# Dashboard Section
# --------------------------------------------------

st.divider()

st.subheader("📊 Expense Analytics Dashboard")


if expenses.empty:

    st.info("Add some expenses to see analytics.")

else:

    # --------------------------------------------------
    # Sidebar Filters
    # --------------------------------------------------

    st.sidebar.header("🔎 Filters")

    categories = ["All"] + sorted(
        expenses["Category"].dropna().unique().tolist()
    )

    selected_category = st.sidebar.selectbox(
        "Select Category",
        categories
    )

    min_date = expenses["Date"].min().date()
    max_date = expenses["Date"].max().date()

    selected_dates = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # --------------------------------------------------
    # Apply Filters
    # --------------------------------------------------

    filtered_expenses = expenses.copy()

    if selected_category != "All":

        filtered_expenses = filtered_expenses[
            filtered_expenses["Category"] == selected_category
        ]

    if len(selected_dates) == 2:

        start_date, end_date = selected_dates

        filtered_expenses = filtered_expenses[
            (filtered_expenses["Date"].dt.date >= start_date)
            &
            (filtered_expenses["Date"].dt.date <= end_date)
        ]

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    total_spending = filtered_expenses["Amount"].sum()

    total_transactions = len(filtered_expenses)

    average_expense = (
        filtered_expenses["Amount"].mean()
        if total_transactions > 0
        else 0
    )

    if total_transactions > 0:

        top_category = (
            filtered_expenses
            .groupby("Category")["Amount"]
            .sum()
            .idxmax()
        )

    else:

        top_category = "N/A"

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "💰 Total Spending",
            f"₹{total_spending:,.2f}"
        )

    with col2:

        st.metric(
            "🧾 Transactions",
            total_transactions
        )

    with col3:

        st.metric(
            "📌 Average Expense",
            f"₹{average_expense:,.2f}"
        )

    with col4:

        st.metric(
            "🏆 Top Category",
            top_category
        )

    # --------------------------------------------------
    # Budget Alert
    # --------------------------------------------------

    st.divider()

    st.subheader("💰 Budget Management")

    budget_progress = min(
        total_spending / monthly_budget,
        1.0
    ) if monthly_budget > 0 else 0

    st.progress(budget_progress)

    st.write(
        f"Spent: ₹{total_spending:,.2f} / "
        f"Budget: ₹{monthly_budget:,.2f}"
    )

    if monthly_budget > 0 and total_spending > monthly_budget:

        st.error("⚠️ Budget exceeded!")

    elif monthly_budget > 0 and total_spending >= monthly_budget * 0.8:

        st.warning("⚠️ You have used more than 80% of your budget.")

    else:

        st.success("✅ You are within your budget.")

    # --------------------------------------------------
    # Charts
    # --------------------------------------------------

    if not filtered_expenses.empty:

        category_summary = (
            filtered_expenses
            .groupby("Category", as_index=False)["Amount"]
            .sum()
            .sort_values("Amount", ascending=False)
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📊 Category-wise Spending")

            fig_bar = px.bar(
                category_summary,
                x="Category",
                y="Amount",
                title="Spending by Category",
                text_auto=".2f"
            )

            st.plotly_chart(
                fig_bar,
                use_container_width=True
            )

        with col2:

            st.subheader("🥧 Spending Distribution")

            fig_pie = px.pie(
                category_summary,
                names="Category",
                values="Amount",
                title="Category Distribution"
            )

            st.plotly_chart(
                fig_pie,
                use_container_width=True
            )

        # --------------------------------------------------
        # Monthly Trend
        # --------------------------------------------------

        st.subheader("📈 Monthly Spending Trend")

        monthly_summary = (
            filtered_expenses
            .assign(
                Month=filtered_expenses["Date"]
                .dt.to_period("M")
                .astype(str)
            )
            .groupby("Month", as_index=False)["Amount"]
            .sum()
        )

        fig_line = px.line(
            monthly_summary,
            x="Month",
            y="Amount",
            markers=True,
            title="Monthly Spending"
        )

        st.plotly_chart(
            fig_line,
            use_container_width=True
        )

    else:

        st.warning("No expenses found for selected filters.")


# --------------------------------------------------
# Expense History
# --------------------------------------------------

st.divider()

st.subheader("📋 Expense History")

if expenses.empty:

    st.info("No expenses added yet.")

else:

    display_expenses = expenses.sort_values(
        "Date",
        ascending=False
    ).reset_index()

    st.dataframe(
        display_expenses,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # Delete Expense
    # --------------------------------------------------

    st.subheader("🗑️ Delete Expense")

    delete_index = st.number_input(
        "Enter original row index to delete",
        min_value=0,
        max_value=len(expenses) - 1,
        step=1
    )

    if st.button("🗑️ Delete Selected Expense"):

        delete_expense(delete_index)

        st.success("Expense deleted successfully!")

        st.rerun()

    # --------------------------------------------------
    # Export CSV
    # --------------------------------------------------

    st.subheader("📥 Export Expense Report")

    csv_data = expenses.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download CSV Report",
        data=csv_data,
        file_name="expense_report.csv",
        mime="text/csv"
    )