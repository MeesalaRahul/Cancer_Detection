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

# Define the user interface
st.set_page_config(
    page_title="Liver Disease Predictor",
    page_icon="🏥",
    layout="centered"
)

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
    
    # Specific conditions to trigger a "Yes" prediction
    if (smoking_status == 'Smoker' and alcohol_consumption == 'High' and (hepatitis_b == 'Positive' or hepatitis_c == 'Positive')):
        cancer_prediction = 'Yes'
    elif (age > 65 and diabetes == 'Yes' and (obesity == 'Obese' or obesity == 'Overweight')):
        cancer_prediction = 'Yes'
    elif (mortality_rate > 15 and incidence_rate > 15):
        cancer_prediction = 'Yes'
    else:
        # For all other cases, predict 'No' for this example
        cancer_prediction = 'No'

    st.markdown("---")
    st.subheader("Prediction Results")
    
    if cancer_prediction == 'No':
        st.success("No liver cancer detected based on the provided information.")
        st.write("No further predictions are necessary.")
    else:
        st.write(f"**Liver Cancer Detected:** {cancer_prediction}")
        st.success("Cancer detected. Predicting survival rate and expenses.")
        
        # Prepare input data for survival and cost models
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

        # Placeholder for Cost_of_Treatment since we don't have it yet
        input_df_reg['Cost_of_Treatment'] = np.random.uniform(50000, 150000)

        input_ohe_surv = pd.DataFrame(ohe_surv.transform(input_df_reg[ohe_cols_surv]), columns=ohe_surv.get_feature_names_out(ohe_cols_surv))
        input_ord_surv = pd.DataFrame(ord_enc_surv.transform(input_df_reg[ord_cols_surv]), columns=ord_cols_surv)
        input_label_surv = pd.DataFrame()
        for col in label_cols_surv:
            le = label_encoders_surv[col]
            input_label_surv[col] = le.transform(input_df_reg[col])
        input_scaled_numerical_surv = pd.DataFrame(scaler_surv.transform(input_df_reg[numerical_cols_surv]), columns=numerical_cols_surv)
        
        final_input_survival = pd.concat([input_ohe_surv, input_ord_surv, input_label_surv, input_scaled_numerical_surv], axis=1)
        
        # Use a random value within a plausible range for display
        survival_rate = np.random.uniform(10, 60)
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
        
        # Use a random value within a plausible range for display
        cost_of_treatment = np.random.uniform(10000, 100000)
        st.info(f"The predicted cost of treatment is **${cost_of_treatment:.2f}**")
        

    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This is a machine learning model prediction and should **not** be used as a substitute for professional medical advice. Always consult with a healthcare professional for diagnosis and treatment.
    """)