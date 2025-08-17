import streamlit as st
import pandas as pd
import pickle

# Load the pipeline
with open("rf_fraud_detection.pkl", "rb") as f:
    model = pickle.load(f)

# App title
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("Fraud Detection Prediction App")
st.markdown(
    """
    Enter the transaction details below and click **Predict** to check if the transaction may be fraudulent.  
    """
)

# Sidebar info
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        """
        This app uses a trained machine learning model to predict the likelihood of a fraudulent transaction.  
        Please provide realistic transaction details for best results.
        """
    )

# Input form
with st.form("transaction_form"):
    st.subheader("Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox(
            "Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
        )
        amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
        oldBalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=1000.0, step=100.0)

    with col2:
        newBalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0, step=100.0)
        oldBalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0, step=100.0)
        newBalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0, step=100.0)

    submitted = st.form_submit_button("🚀 Predict")

if submitted:
    input_data = pd.DataFrame(
        [{
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldBalanceOrg,
            "newbalanceOrig": newBalanceOrig,
            "oldbalanceDest": oldBalanceDest,
            "newbalanceDest": newBalanceDest
        }]
    )

    prediction = model.predict(input_data)[0]

    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error("⚠️ This transaction is **likely FRAUDULENT**.")
    else:
        st.success("✅ This transaction appears **safe**.")

    # Show input summary
    with st.expander("🔍 View Submitted Details"):
        st.table(input_data)

