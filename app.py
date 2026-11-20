
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict whether a transaction is fraudulent.")


@st.cache_resource
def load_model():
    return joblib.load(
        Path(__file__).parent / "credit_card_fraud_model.pkl"
    )


@st.cache_data
def load_defaults():
    return pd.read_csv(
        Path(__file__).parent / "credit_card_fraud_drift_2026.csv"
    )


try:
    model = load_model()
    reference_data = load_defaults()
    features = list(model.feature_names_in_)

except Exception as e:
    st.error(f"Could not load model or dataset: {e}")
    st.stop()


st.subheader("Transaction Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=5000.0
        )

        hour = st.slider("Transaction Hour", 0, 23, 14)

        age = st.number_input(
            "Customer Age",
            min_value=18,
            max_value=100,
            value=25
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=720
        )

        merchant = st.selectbox(
            "Merchant Category",
            sorted(reference_data["merchant_category"].dropna().unique())
        )

        country = st.selectbox(
            "Merchant Country",
            sorted(reference_data["merchant_country"].dropna().unique())
        )

        channel = st.selectbox(
            "Transaction Channel",
            sorted(reference_data["transaction_channel"].dropna().unique())
        )

        transaction_day = st.selectbox(
            "Transaction Day",
            sorted(reference_data["transaction_day_of_week"].dropna().unique())
        )

    with col2:

        customer_country = st.selectbox(
            "Customer Country",
            sorted(reference_data["customer_country"].dropna().unique())
        )

        card_present = st.selectbox("Card Present", [0, 1])

        international = st.selectbox(
            "International Transaction", [0, 1]
        )

        high_risk = st.selectbox("High Risk Country", [0, 1])

        used_chip = st.selectbox("Used Chip", [0, 1])

        used_pin = st.selectbox("Used PIN", [0, 1])

        transactions_24h = st.number_input(
            "Transactions in Last 24 Hours",
            min_value=0,
            value=2
        )

    submitted = st.form_submit_button(
        "Predict Fraud",
        type="primary",
        use_container_width=True
    )


if submitted:

    try:

        # Create baseline values for all model features
        input_data = reference_data[features].iloc[[0]].copy()

        for col in features:

            if pd.api.types.is_numeric_dtype(reference_data[col]):
                input_data[col] = reference_data[col].median()

            else:
                input_data[col] = reference_data[col].mode(
                    dropna=True
                ).iloc[0]

        # Apply user inputs
        input_data["transaction_amount"] = amount
        input_data["transaction_hour"] = hour
        input_data["customer_age"] = age
        input_data["credit_score"] = credit_score
        input_data["merchant_category"] = merchant
        input_data["merchant_country"] = country
        input_data["transaction_channel"] = channel
        input_data["transaction_day_of_week"] = transaction_day
        input_data["customer_country"] = customer_country
        input_data["card_present"] = card_present
        input_data["international_transaction"] = international
        input_data["high_risk_country"] = high_risk
        input_data["used_chip"] = used_chip
        input_data["used_pin"] = used_pin
        input_data["transactions_last_24h"] = transactions_24h

        # Prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        fraud_index = list(model.classes_).index(1)
        fraud_probability = probabilities[fraud_index] * 100

        # Display result
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("🚨 Fraudulent Transaction")
        else:
            st.success("✅ Legitimate Transaction")

        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

        st.progress(float(fraud_probability / 100))

        st.caption(
            "This is a demonstration prediction, not a guarantee. "
            "15 features are entered by the user; the remaining "
            "30 features use dataset-based baseline values."
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")


