import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

#App configuration
st.set_page_config(page_title="AML Fraud Detector", layout="wide")

st.title("🚨 AML – Suspicious Transaction Detection")

def predict_with_api(transaction: dict):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=transaction)
        if response.status_code == 200:
            return response.json()["prediction"]
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ API connection error: {e}"

# Load the model and scaler
try:
    flags = pd.read_csv("data/flags.csv")
    st.toast("✅ Loaded flagged transactions", icon="✅")

    # Prepare data for supervised model
    features = ["amount", "transaction_type", "country", "hour"]
    X = flags[features].copy()
    y = flags["user_flag"].map({"ok": 0, "fraud": 1})  # binary classification 

    # One-hot encoding
    X = pd.get_dummies(X, columns=["transaction_type", "country"], drop_first=True)

    # Scaling
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X[["amount", "hour"]] = scaler.fit_transform(X[["amount", "hour"]])

    # Flag if training data is ready
    training_data_ready = True

except Exception as e:
    st.warning(f"⚠️ Failed to prepare the supervised model: {e}")
    training_data_ready = False

from sklearn.ensemble import RandomForestClassifier

if training_data_ready:
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, y)
    st.toast("🎓 RandomForest model trained on your flagged transactions!", icon="🎓")


# Load data
st.markdown("### 📂 Upload your own CSV file (optional)")

uploaded_file = st.file_uploader("Select a CSV file with transaction data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.toast("📁 Custom CSV file loaded successfully!", icon="📁")
else:
    df = pd.read_csv("data/transactions.csv")
    st.toast("ℹ️ Using default file: `transactions.csv`", icon="ℹ️")

import numpy as np

# Add a customer_id column for demonstration puropses
np.random.seed(42)  # for reproducibility
df["customer_id"] = np.random.randint(10000, 10050, size=len(df))

# Display basic info about dataset
if st.checkbox("📊 Show transaction data"):
    st.dataframe(df.head(20))

# Preprocessing
features = ["amount", "transaction_type", "country", "hour"]
df_model = df[features].copy()

# Categorical variable encoding
df_model = pd.get_dummies(df_model, columns=["transaction_type", "country"], drop_first=True)

# Scaling
scaler = StandardScaler()
df_model[["amount", "hour"]] = scaler.fit_transform(df_model[["amount", "hour"]])

from sklearn.svm import OneClassSVM

# Choose model for fraud detection
st.markdown("## 🤖 Select a fraud detection model")

model_choice = st.selectbox("🔍 Detection model", ["Isolation Forest", "One-Class SVM"])

if model_choice == "Isolation Forest":
    contamination = st.slider("🎚️ Procent podejrzanych (contamination)", 0.01, 0.2, 0.05, step=0.01)
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(df_model)
    pred = model.predict(df_model)
    df["fraud_flag"] = [1 if p == -1 else 0 for p in pred]

elif model_choice == "One-Class SVM":
    nu = st.slider("🎚️ Parametr nu", 0.01, 0.2, 0.05, step=0.01)
    model = OneClassSVM(kernel="rbf", gamma='scale', nu=nu)
    model.fit(df_model)
    pred = model.predict(df_model)
    df["fraud_flag"] = [1 if p == -1 else 0 for p in pred]

elif model_choice == "RandomForest (Supervised)":
    if not training_data_ready:
        st.warning("⚠️ Brak danych `flags.csv` – oznacz transakcje, by trenować model nadzorowany.")
    else:
        # Prepare the data as before (scaler and one-hot)
        new_data = df[["amount", "transaction_type", "country", "hour"]].copy()
        new_data = pd.get_dummies(new_data, columns=["transaction_type", "country"], drop_first=True)
        # Adjust missing columns
        missing_cols = set(X.columns) - set(new_data.columns)
        for col in missing_cols:
            new_data[col] = 0
        new_data = new_data[X.columns]  # sort columns
        new_data[["amount", "hour"]] = scaler.transform(new_data[["amount", "hour"]])
        
        pred = clf.predict(new_data)
        df["fraud_flag"] = pred

# Save predictions
df["fraud_flag"] = [1 if p == -1 else 0 for p in pred]

# Display only fraud transactions
only_fraud = st.checkbox("🔍 Show only suspicious transactions", value=True)

if only_fraud:
    flagged = df[df["fraud_flag"] == 1]
else:
    flagged = df

# 🔍 Filtering
st.markdown("## 🔍 Transaction filtering")

available_countries = flagged["country"].unique().tolist()
selected_countries = st.multiselect("🌍 Choose country", available_countries, default=available_countries)

available_types = flagged["transaction_type"].unique().tolist()
selected_types = st.multiselect("💳 Transaction type", available_types, default=available_types)

min_amount, max_amount = float(flagged["amount"].min()), float(flagged["amount"].max())
amount_range = st.slider("💰 Amount range", min_value=min_amount, max_value=max_amount,
                         value=(min_amount, max_amount))

filtered = flagged[
    (flagged["country"].isin(selected_countries)) &
    (flagged["transaction_type"].isin(selected_types)) &
    (flagged["amount"].between(*amount_range))
]

# 📊 Results
st.subheader(f"Detected {df['fraud_flag'].sum()} suspicious transactions")
st.dataframe(filtered)

# 📥 Download data
st.download_button("📥 Download suspicious as CSV", data=filtered.to_csv(index=False),
                   file_name="suspicious_transactions.csv", mime="text/csv")

import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("## 📈 Visualizations of suspicious transactions")

if filtered.empty:
    st.warning("No data available to display charts. Please adjust the filters above.")
else:
    col1, col2 = st.columns(2)

    # Amount histogram
    with col1:
        st.markdown("### 💰 Transaction amount histogram")
        fig1, ax1 = plt.subplots()
        sns.histplot(filtered["amount"], bins=30, kde=True, ax=ax1)
        ax1.set_xlabel("Amount")
        ax1.set_ylabel("Number of transactions")
        st.pyplot(fig1)

    # Fraud by country
    with col2:
        st.markdown("### 🌍 Fraud by country")
        fig2, ax2 = plt.subplots()
        country_counts = filtered["country"].value_counts()
        sns.barplot(x=country_counts.values, y=country_counts.index, ax=ax2)
        ax2.set_xlabel("Number of frauds")
        ax2.set_ylabel("Country")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    # Fraud by hour
    with col3:
        st.markdown("### ⏰ Fraud by hour")
        fig3, ax3 = plt.subplots()
        sns.countplot(x="hour", data=filtered, ax=ax3)
        ax3.set_xlabel("Transaction hour")
        ax3.set_ylabel("Number of frauds")
        st.pyplot(fig3)

    # Fraud by transaction type
    with col4:
        st.markdown("### 💳 Fraud by transaction type")
        fig4, ax4 = plt.subplots()
        type_counts = filtered["transaction_type"].value_counts()
        sns.barplot(x=type_counts.values, y=type_counts.index, ax=ax4)
        ax4.set_xlabel("Number of frauds")
        ax4.set_ylabel("Transaction type")
        st.pyplot(fig4)

st.markdown("## 👤 Suspicious Clients Dashboard")

# Group by clients
client_summary = filtered.groupby("customer_id").agg(
    num_frauds=("fraud_flag", "sum"),
    total_amount=("amount", "sum"),
    avg_amount=("amount", "mean"),
    num_transactions=("amount", "count")
).sort_values(by="num_frauds", ascending=False)

st.dataframe(client_summary)

# Select a client
selected_client = st.selectbox("👤 Select client", options=["-- Select a client --"] + list(client_summary.index))

# Show their transactions only if a client is selected
if selected_client != "-- Select a client --":
    st.markdown(f"### 🧾 Transactions for client `{selected_client}`")
    client_transactions = filtered[filtered["customer_id"] == selected_client]
    st.dataframe(client_transactions)

st.markdown("## 🏷️ Manual Fraud Tagging")

tx_options = ["-- Select a transaction --"] + filtered.index.astype(str).tolist()
selected_tx = st.selectbox("🔍 Choose transaction ID to review", tx_options)

if selected_tx != "-- Select a transaction --":
    selected_row = filtered.loc[int(selected_tx)]
    st.write("### Transaction details:")
    st.json(selected_row.to_dict())

    decision = st.radio("📝 Label this transaction as:", ["✅ Legit", "🚩 Fraud"], horizontal=True)

    if st.button("💾 Save label"):
        feedback_row = selected_row.copy()
        feedback_row["user_flag"] = "fraud" if decision == "🚩 Fraud" else "ok"

        try:
            existing_flags = pd.read_csv("data/flags.csv")
            updated_flags = pd.concat([existing_flags, feedback_row.to_frame().T], ignore_index=True)
        except FileNotFoundError:
            updated_flags = feedback_row.to_frame().T

        updated_flags.to_csv("data/flags.csv", index=False)
        st.toast("✅ Label saved to `flags.csv`", icon="✅")

st.markdown("## 🗂️ Your Flagged Transactions")

try:
    flags = pd.read_csv("data/flags.csv")

    # Filtering option
    view_option = st.radio("📂 Show flagged as:", ["All", "✅ Legit", "🚩 Fraud"], horizontal=True)

    if view_option == "✅ Legit":
        view_flags = flags[flags["user_flag"] == "ok"]
    elif view_option == "🚩 Fraud":
        view_flags = flags[flags["user_flag"] == "fraud"]
    else:
        view_flags = flags

    st.dataframe(view_flags)

    # Simple summary chart
    st.markdown("### 📊 Number of labels:")
    label_counts = flags["user_flag"].value_counts()
    st.bar_chart(label_counts)

except FileNotFoundError:
    st.toast("📭 You haven't flagged any transactions yet", icon="📭")

st.markdown("## 🤖 Quick Prediction via API")

with st.form("predict_form"):
    amount = st.number_input("Transaction amount", min_value=0.0, step=1.0)
    hour = st.slider("Hour (0–23)", 0, 23, 12)
    transaction_type = st.selectbox("Transaction type", ["transfer", "withdrawal"])
    country = st.selectbox("Country", ["PL", "DE", "UA"])

    submitted = st.form_submit_button("🔍 Check transaction")

    if submitted:
        tx = {
            "amount": amount,
            "hour": hour,
            "transaction_type": transaction_type,
            "country": country
        }
        result = predict_with_api(tx)
        st.toast(f"📤 Model response: **{result.upper()}**", icon="📤")
