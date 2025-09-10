import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load all trained models and preprocessing objects
try:
    survival_regressor = joblib.load('survival_regressor.joblib')
    cost_regressor = joblib.load('cost_regressor.joblib')
    
    ohe_surv = joblib.load('ohe_surv.joblib')
    ord_enc_surv = joblib.load('ord_enc_surv.joblib')
    scaler_surv = joblib.load('scaler_surv.joblib')
    label_encoders_surv = joblib.load('label_encoders_surv.joblib')

    ohe_cost = joblib.load('ohe_cost.joblib')
    ord_enc_cost = joblib.load('ord_enc_cost.joblib')
    scaler_cost = joblib.load('scaler_cost.joblib')
    label_encoders_cost = joblib.load('label_encoders_cost.joblib')

except FileNotFoundError:
    st.error("Error: Regression model or preprocessing files not found. Please run `train_model.py` first.")
    st.stop()

# Use custom CSS for an attractive interface
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f0f2f5;
    }
    .main {
        padding: 20px;
    }
    .container {
        max-width: 900px;
        margin: auto;
        padding: 40px;
        background-color: #fff;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    header {
        text-align: center;
        margin-bottom: 30px;
    }
    header h1 {
        font-weight: 600;
        color: #1a237e;
        font-size: 2.5rem;
        margin-bottom: 5px;
    }
    header p {
        color: #555;
        font-size: 1rem;
    }
    .form-container, .results-container {
        background-color: #f9f9f9;
        padding: 30px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    form h2 {
        color: #1a237e;
        margin-top: 0;
        margin-bottom: 20px;
        font-size: 1.5rem;
    }
    .input-row {
        display: flex;
        gap: 20px;
        margin-bottom: 15px;
    }
    .stSelectbox, .stNumberInput {
        flex: 1;
    }
    button {
        width: 100%;
        padding: 15px;
        background-color: #43a047;
        color: white;
        border: none;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.3s ease;
        margin-top: 20px;
    }
    button:hover {
        background-color: #388e3c;
    }
    .results-container {
        text-align: center;
    }
    .result-card {
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .result-success {
        background-color: #e8f5e9;
        border: 1px solid #43a047;
        color: #2e7d32;
    }
    .result-warning {
        background-color: #fff8e1;
        border: 1px solid #ffb300;
        color: #f57f17;
    }
    .result-error {
        background-color: #ffebee;
        border: 1px solid #e53935;
        color: #c62828;
    }
    h3.result-title {
        margin-top: 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .result-text {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Define the user interface
st.title("🏥 Liver Disease Predictor")
st.markdown("""
This application detects the presence of liver cancer based on specific conditions. If cancer is detected, it then predicts the patient's survival rate and estimated treatment costs.
""")
st.markdown("---")

# Input fields for user data
with st.form("prediction_form"):
    st.header("Patient Information")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (in years)", min_value=20, max_value=90, value=50)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        smoking_status = st.selectbox("Smoking Status", options=["Smoker", "Non-Smoker"])
        rural_urban = st.selectbox("Rural or Urban", options=["Rural", "Urban"])
        
    with col2:
        alcohol_consumption = st.selectbox("Alcohol Consumption", options=["Low", "Moderate", "High"])
        obesity = st.selectbox("Obesity Status", options=["Underweight", "Normal", "Overweight", "Obese"])
        seafood_consumption = st.selectbox("Seafood Consumption", options=["Low", "Medium", "High"])
        diabetes = st.selectbox("Diabetes", options=["Yes", "No"])
        
    col3, col4 = st.columns(2)
    with col3:
        hepatitis_b = st.selectbox("Hepatitis B Status", options=["Positive", "Negative"])
        hepatitis_c = st.selectbox("Hepatitis C Status", options=["Positive", "Negative"])
        herbal_medicine = st.selectbox("Herbal Medicine Use", options=["Yes", "No"])

    st.subheader("Disease & Healthcare Metrics")
    col5, col6, col7 = st.columns(3)
    with col5:
        incidence_rate = st.number_input("Incidence Rate", min_value=0.0, value=9.0)
    with col6:
        mortality_rate = st.number_input("Mortality Rate", min_value=0.0, value=10.0)
    with col7:
        healthcare_access = st.selectbox("Healthcare Access", options=["Poor", "Moderate", "Good"])
    
    col8, col9 = st.columns(2)
    with col8:
        preventive_care = st.selectbox("Preventive Care", options=["Poor", "Moderate", "Good"])
    with col9:
        pass

    submitted = st.form_submit_button("Get Prediction")

if submitted:
    # --- Step 1: Detect Cancer using rule-based system ---
    cancer_prediction = 'No'
    
    if (smoking_status == 'Smoker' and alcohol_consumption == 'High' and (hepatitis_b == 'Positive' or hepatitis_c == 'Positive')):
        cancer_prediction = 'Yes'
    elif (age > 65 and diabetes == 'Yes' and (obesity == 'Obese' or obesity == 'Overweight')):
        cancer_prediction = 'Yes'
    elif (mortality_rate > 15 and incidence_rate > 15):
        cancer_prediction = 'Yes'
    else:
        cancer_prediction = 'No'

    st.markdown("---")
    st.subheader("Prediction Results")
    
    if cancer_prediction == 'No':
        st.success("No liver cancer detected based on the provided information.")
    else:
        st.write(f"**Liver Cancer Detected:** {cancer_prediction}")
        st.success("Cancer detected. Predicting survival rate and expenses.")
        
        input_data_reg = {
            'Gender': gender, 'Age': age, 'Alcohol_Consumption': alcohol_consumption,
            'Smoking_Status': smoking_status, 'Hepatitis_B_Status': hepatitis_b,
            'Hepatitis_C_Status': hepatitis_c, 'Obesity': obesity, 'Diabetes': diabetes,
            'Rural_or_Urban': rural_urban, 'Seafood_Consumption': seafood_consumption,
            'Herbal_Medicine_Use': herbal_medicine, 'Incidence_Rate': incidence_rate,
            'Mortality_Rate': mortality_rate, 'Healthcare_Access': healthcare_access,
            'Preventive_Care': preventive_care
        }
        input_df_reg = pd.DataFrame([input_data_reg])

        # --- Step 2: Predict Survival Rate ---
        # The Survival Regressor uses its own preprocessing
        ohe_cols_surv = ohe_surv.feature_names_in_
        ord_cols_surv = ord_enc_surv.feature_names_in_
        label_cols_surv = label_encoders_surv.keys()
        numerical_cols_surv = ['Age', 'Incidence_Rate', 'Mortality_Rate', 'Cost_of_Treatment']

        input_df_reg['Cost_of_Treatment'] = 0.0 # Placeholder
        
        input_ohe_surv = pd.DataFrame(ohe_surv.transform(input_df_reg[ohe_cols_surv]), columns=ohe_surv.get_feature_names_out(ohe_cols_surv))
        input_ord_surv = pd.DataFrame(ord_enc_surv.transform(input_df_reg[ord_cols_surv]), columns=ord_cols_surv)
        input_label_surv = pd.DataFrame()
        for col in label_cols_surv:
            le = label_encoders_surv[col]
            input_label_surv[col] = le.transform(input_df_reg[col])
        input_scaled_numerical_surv = pd.DataFrame(scaler_surv.transform(input_df_reg[numerical_cols_surv]), columns=numerical_cols_surv)
        
        final_input_survival = pd.concat([input_ohe_surv, input_ord_surv, input_label_surv, input_scaled_numerical_surv], axis=1)
        
        survival_rate = survival_regressor.predict(final_input_survival)[0]
        st.info(f"The predicted survival rate is **{survival_rate:.2f}%**")
        
        if survival_rate >= 70:
            st.success("The patient has a **high** predicted survival rate.")
        elif survival_rate >= 40:
            st.warning("The patient has a **moderate** predicted survival rate.")
        else:
            st.error("The patient has a **low** predicted survival rate.")

        # --- Step 3: Predict Expenses ---
        # The Cost Regressor uses its own preprocessing
        ohe_cols_cost = ohe_cost.feature_names_in_
        ord_cols_cost = ord_enc_cost.feature_names_in_
        label_cols_cost = label_encoders_cost.keys()
        numerical_cols_cost = ['Age', 'Incidence_Rate', 'Mortality_Rate', 'Survival_Rate']

        input_df_cost = pd.DataFrame({**input_data_reg, 'Survival_Rate': survival_rate}, index=[0])

        input_ohe_cost = pd.DataFrame(ohe_cost.transform(input_df_cost[ohe_cols_cost]), columns=ohe_cost.get_feature_names_out(ohe_cols_cost))
        input_ord_cost = pd.DataFrame(ord_enc_cost.transform(input_df_cost[ord_cols_cost]), columns=ord_cols_cost)
        input_label_cost = pd.DataFrame()
        for col in label_cols_cost:
            le = label_encoders_cost[col]
            input_label_cost[col] = le.transform(input_df_cost[col])
        input_scaled_numerical_cost = pd.DataFrame(scaler_cost.transform(input_df_cost[numerical_cols_cost]), columns=numerical_cols_cost)

        final_input_cost = pd.concat([input_ohe_cost, input_ord_cost, input_label_cost, input_scaled_numerical_cost], axis=1)
        cost_of_treatment = cost_regressor.predict(final_input_cost)[0]
        
        st.info(f"The predicted cost of treatment is **${cost_of_treatment:.2f}**")
        
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This is a machine learning model prediction and should **not** be used as a substitute for professional medical advice. Always consult with a healthcare professional for diagnosis and treatment.
    """)
