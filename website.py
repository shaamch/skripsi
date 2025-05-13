import streamlit as st
import pandas as pd
import pickle
import os

# ==================== Load CSS Style ====================
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==================== Judul dan Deskripsi ====================
st.title("💳 Aplikasi Prediksi Penipuan Transaksi Digital")


# ==================== Load Model, Encoder, Scaler ====================
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

# ==================== Load Data for Reference ====================
@st.cache_data
def load_data():
    file_path = "fraudTrain_dataset_cleaned.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        st.error("Dataset tidak ditemukan.")
        return None

df = load_data()

if df is not None:
    target_col = 'is_fraud'
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    X = df.drop(columns=[target_col])

    # ==================== Formulir Input ====================
    st.subheader("📋 Masukan Data Transaksi")
    st.markdown("Lengkapi informasi transaksi di bawah ini untuk mendapatkan hasil prediksi.")

    with st.form("fraud_form"):
        user_input = {}
        for col in X.columns:
            if col in categorical_cols:
                user_input[col] = st.selectbox(f"{col}", df[col].dropna().unique())
            else:
                user_input[col] = st.number_input(f"{col}", value=float(df[col].median()))
        submitted = st.form_submit_button("🔍 Prediksi")

    # ==================== Prediksi ====================
    if submitted:
        input_df = pd.DataFrame([user_input])
        input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        st.subheader("📢 Hasil Prediksi")
        if prediction == 1:
            st.error(f"🚨 Transaksi ini **berpotensi penipuan**!\n\nProbabilitas: **{prob:.2%}**")
        else:
            st.success(f"✅ Transaksi ini **aman**.\n\nProbabilitas penipuan: **{prob:.2%}**")

else:
    st.stop()
