import streamlit as st
import pandas as pd
import pickle

st.title("💳 Prediksi Transaksi Penipuan (Fraud Detection)")

st.markdown("Gunakan aplikasi ini untuk memprediksi apakah suatu transaksi berisiko penipuan.")

# Load model, encoder, scaler
@st.cache_resource
def load_model():
    with open("xgboost_fraud_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("ordinal_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("fraud_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, encoder, scaler

model, encoder, scaler = load_model()

# Load sample dataset hanya untuk referensi kolom & nilai default
@st.cache_data
def load_data():
    return pd.read_csv("Clean Dataset/fraudTrain_dataset_cleaned.csv")

df = load_data()

# Tentukan kolom target dan kategorikal
target_col = 'is_fraud'
categorical_cols = df.select_dtypes(include='object').columns.tolist()
X = df.drop(columns=[target_col])

# Formulir input dari user
st.subheader("📋 Formulir Data Transaksi")
with st.form("fraud_form"):
    user_input = {}
    for col in X.columns:
        if col in categorical_cols:
            user_input[col] = st.selectbox(f"{col}", df[col].dropna().unique())
        else:
            user_input[col] = st.number_input(f"{col}", value=float(df[col].median()))
    submitted = st.form_submit_button("🔍 Prediksi")

# Prediksi
if submitted:
    input_df = pd.DataFrame([user_input])
    input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 Transaksi ini berpotensi penipuan! (Probabilitas: {prob:.2%})")
    else:
        st.success(f"✅ Transaksi ini aman. (Probabilitas penipuan: {prob:.2%})")
