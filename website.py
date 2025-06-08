import streamlit as st
import pandas as pd 
import pickle
import os
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# ==================== Theme Toggle & Load CSS ====================
theme = st.sidebar.radio("Theme", ["🌞Light", "🌚Dark"], horizontal=True)

if theme == "🌞Light":
    css_file = "style_light.css"
else:
    css_file = "style_dark.css"

with open(css_file) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==================== Label Mapping ====================
column_labels = {
    "amt": "Amount",
    "gender": "Gender",
    "state": "State",
    "city": "City",
    "city_pop": "City Population",
    "job": "Job",
    "category": "Merchant Category",
    "street": "Street",
    "zip": "ZIP Code",
    "age": "Customer Age",
    "day_of_week": "Day of Week",
    "transaction_min": "Transaction Minute",
    "transaction_hour": "Transaction Hour",
    "transaction_date": "Transaction Date",
    "transaction_month": "Transaction Month",
    "transaction_distance": "Transaction Distance"
}

# ==================== Title & Description ====================
st.title("💳 Fraud Transaction Detection App")
st.markdown("""
This application helps you predict whether a transaction is **safe** or **potentially fraudulent** using machine learning.
""")

# ==================== Sidebar Instructions ====================
with st.sidebar:
    st.header("📘 How to Use")
    st.markdown("""
1. Fill in all transaction details in the form.
2. Click **Predict** to get the result.
3. The system will tell you if the transaction looks suspicious.

---

⚠️ **Be careful!** Fraudulent transactions often:
- Happen at unusual times.
- Involve distant locations.
- Use large/unusual amounts.

This app is for educational and awareness purposes only.

---
""")

# ==================== Input Descriptions ====================
st.markdown("""<h4>📝 Input Descriptions</h4>
Below is a list of all input fields used in the prediction. Fields marked with 
<span style='color:green; font-weight:bold'>(Required)</span> or 
<span style='color:orange; font-weight:bold'>(Optional)</span>.
<ul>
  <li><b>Amount (amt)</b> – Transaction amount <span style='color:green'><b>(Required)</b></span></li>
  <li><b>Gender (gender)</b> – Customer gender <span style='color:green'><b>(Required)</b></span></li>
  <li><b>State (state)</b> – Location of transaction <span style='color:green'><b>(Required)</b></span></li>
  <li><b>City (city)</b> – City of transaction <span style='color:green'><b>(Required)</b></span></li>
  <li><b>City Population (city_pop)</b> – Population of the city <span style='color:orange'><b>(Optional)</b></span></li>
  <li><b>Job (job)</b> – Customer's occupation <span style='color:orange'><b>(Optional)</b></span></li>
  <li><b>Merchant Category (category)</b> – Type of merchant <span style='color:green'><b>(Required)</b></span></li>
  <li><b>Street (street)</b> – Street name <span style='color:green'><b>(Required)</b></span></li>
  <li><b>ZIP Code (zip)</b> – Postal code <span style='color:green'><b>(Required)</b></span></li>
  <li><b>Customer Age (age)</b> – Age in years <span style='color:green'><b>(Required)</b></span></li>
  <li><b>Day of Week (day_of_week)</b> – Day when the transaction occurred <span style='color:green'><b>(Required)</b></span></li>
  <li><b>Transaction Minute (transaction_min)</b> – Minute within the hour <span style='color:green'><b>(Required)</b></span></li>
  <li><b>Transaction Hour (transaction_hour)</b> – Hour of the day (0–23) <span style='color:green'><b>(Required)</b></span></li>
  <li><b>Transaction Date (transaction_date)</b> – Date of transaction <span style='color:green'><b>(Required)</b></span></li>
  <li><b>Transaction Month (transaction_month)</b> – Month of transaction (1–12) <span style='color:green'><b>(Required)</b></span></li>
  <li><b>Transaction Distance (transaction_distance)</b> – Distance between customer and merchant <span style='color:orange'><b>(Optional)</b></span></li>
</ul>
""", unsafe_allow_html=True)

# ==================== Load Dataset ====================
@st.cache_data
def load_data():
    path = "Clean Dataset/fraudTrain_dataset_beforeRUS.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        st.error("❌ Dataset not found.")
        return None

df = load_data()
if df is None:
    st.stop()

# ==================== Prepare Data & Model ====================
target_col = 'is_fraud'
categorical_cols = df.select_dtypes(include='object').columns.tolist()
X = df.drop(columns=[target_col])
y = df[target_col]

# Fit encoder and scaler
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_encoded = X.copy()
X_encoded[categorical_cols] = encoder.fit_transform(X[categorical_cols])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Train model (could be replaced with loaded one if desired)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# ==================== Save for future use ====================
with open("ordinal_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("fraud_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("xgboost_fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

# ==================== Prediction Form ====================
st.subheader("📋 Transaction Details Form")
st.markdown("Please fill in all fields to get a prediction.")

with st.form("fraud_form"):
    user_input = {}
    for col in X.columns:
        label = column_labels.get(col, col)  # Gunakan label mapping
        if col in categorical_cols:
            options = df[col].dropna().unique().tolist()
            options.insert(0, "None")
            user_input[col] = st.selectbox(f"{label}", options, index=0)
        else:
            user_input[col] = st.number_input(f"{label}", value=0.0)
    submitted = st.form_submit_button("🔍 Predict")

# ==================== Prediction Logic ====================
if submitted:
    input_df = pd.DataFrame([user_input])
    input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("📢 Prediction Result")
    st.markdown("Be more careful and alert, hopefully bad days will not happen to all of us 😊")

    if prediction == 1:
        st.error(f"🚨 This transaction is **potentially fraudulent**!\n\nProbability: **{prob:.2%}**")
    else:
        st.success(f"✅ This transaction appears **safe**.\n\nFraud probability: **{prob:.2%}**")
