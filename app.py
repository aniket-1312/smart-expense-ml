import os
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Smart Expense Tracker",
    page_icon="💰",
    layout="wide"
)


# ==================================================
# CUSTOM UI
# ==================================================

st.markdown("""
<style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 15px;
    }

</style>
""", unsafe_allow_html=True)


# ==================================================
# FILE PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "models" / "expense_model.pkl"
VECTORIZER_FILE = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

DATA_DIR = BASE_DIR / "data"
EXPENSE_FILE = DATA_DIR / "user_expenses.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    return model, vectorizer


model, vectorizer = load_model()


# ==================================================
# CONSTANTS
# ==================================================

CATEGORIES = [
    "Food",
    "Travel",
    "Shopping",
    "EMI",
    "Investment"
]


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def load_expenses():

    if EXPENSE_FILE.exists():

        df = pd.read_csv(EXPENSE_FILE)

        if not df.empty:

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            df["Amount"] = pd.to_numeric(
                df["Amount"],
                errors="coerce"
            )

            df = df.dropna(
                subset=["Date", "Amount"]
            )

        return df

    return pd.DataFrame(
        columns=[
            "Date",
            "Description",
            "Amount",
            "Category"
        ]
    )


def save_expense(
    expense_date,
    description,
    amount,
    category
):

    new_expense = pd.DataFrame({
        "Date": [expense_date],
        "Description": [description],
        "Amount": [amount],
        "Category": [category]
    })

    if EXPENSE_FILE.exists():

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

    text_tfidf = vectorizer.transform(
        [description]
    )

    prediction = model.predict(
        text_tfidf
    )[0]

    probabilities = model.predict_proba(
        text_tfidf
    )

    confidence = probabilities.max()

    return prediction, confidence


def update_expense(
    row_index,
    expense_date,
    description,
    amount,
    category
):

    df = load_expenses()

    df.loc[row_index, "Date"] = pd.to_datetime(
        expense_date
    )

    df.loc[row_index, "Description"] = description

    df.loc[row_index, "Amount"] = amount

    df.loc[row_index, "Category"] = category

    df.to_csv(
        EXPENSE_FILE,
        index=False
    )


def delete_expense(row_index):

    df = load_expenses()

    df = df.drop(
        index=row_index
    ).reset_index(drop=True)

    df.to_csv(
        EXPENSE_FILE,
        index=False
    )


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">💰 Smart Expense Tracker</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">ML-Powered Personal Finance Dashboard</div>',
    unsafe_allow_html=True
)

st.caption(
    "Automatically categorize expenses, track spending, "
    "and understand your financial habits."
)

st.divider()


# ==================================================
# ADD EXPENSE
# ==================================================

st.markdown(
    '<div class="section-title">➕ Add New Expense</div>',
    unsafe_allow_html=True
)

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

        st.warning(
            "Please enter an expense description."
        )

    elif amount <= 0:

        st.warning(
            "Please enter a valid amount."
        )

    else:

        category, confidence = predict_category(
            description
        )

        save_expense(
            expense_date,
            description,
            amount,
            category
        )

        st.success(
            "Expense saved successfully!"
        )

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Predicted Category",
                category
            )

        with result_col2:

            st.metric(
                "Confidence",
                f"{confidence:.2%}"
            )


# ==================================================
# LOAD DATA
# ==================================================

expenses = load_expenses()


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("⚙️ Settings")

monthly_budget = st.sidebar.number_input(
    "Monthly Budget (₹)",
    min_value=0.0,
    value=10000.0,
    step=500.0
)


# ==================================================
# DASHBOARD
# ==================================================

st.divider()

st.markdown(
    '<div class="section-title">📊 Expense Analytics Dashboard</div>',
    unsafe_allow_html=True
)


if expenses.empty:

    st.info(
        "No expenses added yet. Add your first expense above."
    )

else:

    # --------------------------------------------------
    # FILTERS
    # --------------------------------------------------

    st.sidebar.header("🔎 Filters")

    categories = ["All"] + CATEGORIES

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

    filtered_expenses = expenses.copy()

    if selected_category != "All":

        filtered_expenses = filtered_expenses[
            filtered_expenses["Category"] == selected_category
        ]

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

        start_date, end_date = selected_dates

        filtered_expenses = filtered_expenses[
            (
                filtered_expenses["Date"].dt.date
                >= start_date
            )
            &
            (
                filtered_expenses["Date"].dt.date
                <= end_date
            )
        ]

    # --------------------------------------------------
    # METRICS
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

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:

        st.metric(
            "💰 Total Spending",
            f"₹{total_spending:,.2f}"
        )

    with metric_col2:

        st.metric(
            "🧾 Transactions",
            total_transactions
        )

    with metric_col3:

        st.metric(
            "📌 Average Expense",
            f"₹{average_expense:,.2f}"
        )

    with metric_col4:

        st.metric(
            "🏆 Top Category",
            top_category
        )

    # --------------------------------------------------
    # BUDGET
    # --------------------------------------------------

    st.divider()

    st.subheader("💰 Budget Management")

    if monthly_budget > 0:

        budget_progress = min(
            total_spending / monthly_budget,
            1.0
        )

        st.progress(budget_progress)

        remaining_budget = monthly_budget - total_spending

        st.write(
            f"Spent: ₹{total_spending:,.2f} / "
            f"Budget: ₹{monthly_budget:,.2f}"
        )

        if remaining_budget >= 0:

            st.success(
                f"✅ Remaining budget: ₹{remaining_budget:,.2f}"
            )

        else:

            st.error(
                f"⚠️ Budget exceeded by "
                f"₹{abs(remaining_budget):,.2f}"
            )

    # --------------------------------------------------
    # CHARTS
    # --------------------------------------------------

    if not filtered_expenses.empty:

        category_summary = (
            filtered_expenses
            .groupby(
                "Category",
                as_index=False
            )["Amount"]
            .sum()
            .sort_values(
                "Amount",
                ascending=False
            )
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            st.subheader(
                "📊 Category-wise Spending"
            )

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

        with chart_col2:

            st.subheader(
                "🥧 Spending Distribution"
            )

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
        # MONTHLY TREND
        # --------------------------------------------------

        st.subheader(
            "📈 Monthly Spending Trend"
        )

        monthly_summary = (
            filtered_expenses
            .assign(
                Month=filtered_expenses["Date"]
                .dt.to_period("M")
                .astype(str)
            )
            .groupby(
                "Month",
                as_index=False
            )["Amount"]
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

        st.warning(
            "No expenses found for selected filters."
        )


# ==================================================
# EXPENSE HISTORY
# ==================================================

st.divider()

st.markdown(
    '<div class="section-title">📋 Expense History</div>',
    unsafe_allow_html=True
)


if expenses.empty:

    st.info(
        "No expenses added yet."
    )

else:

    display_expenses = expenses.copy()

    display_expenses["Date"] = (
        display_expenses["Date"]
        .dt.strftime("%Y-%m-%d")
    )

    display_expenses["Amount"] = (
        display_expenses["Amount"]
        .map(lambda x: f"₹{x:,.2f}")
    )

    display_expenses = (
        display_expenses
        .sort_values("Date", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(
        display_expenses,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # EDIT / DELETE
    # --------------------------------------------------

    st.subheader("✏️ Manage Expense")

    expense_options = [
        f"{i} | {row['Description']} | "
        f"₹{row['Amount']:,.2f} | {row['Category']}"
        for i, row in expenses.iterrows()
    ]

    selected_expense = st.selectbox(
        "Select an expense",
        expense_options
    )

    selected_index = int(
        selected_expense.split(" | ")[0]
    )

    action_col1, action_col2 = st.columns(2)

    with action_col1:

        edit_button = st.button(
            "✏️ Edit Selected Expense",
            use_container_width=True
        )

    with action_col2:

        delete_button = st.button(
            "🗑️ Delete Selected Expense",
            use_container_width=True
        )

    # --------------------------------------------------
    # DELETE ACTION
    # --------------------------------------------------

    if delete_button:

        delete_expense(selected_index)

        st.success(
            "Expense deleted successfully!"
        )

        st.rerun()

    # --------------------------------------------------
    # EDIT ACTION
    # --------------------------------------------------

    if edit_button:

        selected_row = expenses.loc[selected_index]

        st.subheader("✏️ Edit Expense Details")

        edit_col1, edit_col2, edit_col3 = st.columns(3)

        with edit_col1:

            edit_date = st.date_input(
                "Edit Date",
                value=selected_row["Date"].date()
            )

        with edit_col2:

            edit_description = st.text_input(
                "Edit Description",
                value=selected_row["Description"]
            )

        with edit_col3:

            edit_amount = st.number_input(
                "Edit Amount (₹)",
                min_value=0.0,
                value=float(selected_row["Amount"]),
                step=10.0
            )

        edit_category = st.selectbox(
            "Edit Category",
            CATEGORIES,
            index=(
                CATEGORIES.index(selected_row["Category"])
                if selected_row["Category"] in CATEGORIES
                else 0
            )
        )

        if st.button(
            "💾 Save Changes",
            use_container_width=True
        ):

            if edit_description.strip() == "":

                st.warning(
                    "Description cannot be empty."
                )

            elif edit_amount <= 0:

                st.warning(
                    "Amount must be greater than zero."
                )

            else:

                update_expense(
                    selected_index,
                    edit_date,
                    edit_description,
                    edit_amount,
                    edit_category
                )

                st.success(
                    "Expense updated successfully!"
                )

                st.rerun()

    # --------------------------------------------------
    # EXPORT CSV
    # --------------------------------------------------

    st.subheader("📥 Export Expense Report")

    csv_data = expenses.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download CSV Report",
        data=csv_data,
        file_name="expense_report.csv",
        mime="text/csv"
    )