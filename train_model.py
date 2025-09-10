import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import numpy as np

def train_and_save_regressors(data_path):
    """
    Trains and saves models for survival rate and cost prediction.
    """
    try:
        liver_data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {data_path}. Please check the path.")
        return

    # Drop non-essential columns
    columns_to_drop = ['Country', 'Region', 'Population', 
                       'Screening_Availability', 'Treatment_Availability', 
                       'Liver_Transplant_Access', 'Ethnicity']
    liver_data = liver_data.drop(columns=columns_to_drop, errors='ignore')

    # Define column groups for preprocessing
    onehot_cols = ['Gender', 'Smoking_Status', 'Rural_or_Urban']
    ordinal_cols = ['Alcohol_Consumption', 'Obesity', 'Seafood_Consumption', 
                    'Healthcare_Access', 'Preventive_Care']
    label_cols = ['Hepatitis_B_Status', 'Hepatitis_C_Status', 'Diabetes', 'Herbal_Medicine_Use']

    # Ordinal categories mapping
    ordinal_categories = [
        ['Low', 'Moderate', 'High'],
        ['Underweight', 'Normal', 'Overweight', 'Obese'],
        ['Low', 'Medium', 'High'],
        ['Poor', 'Moderate', 'Good'],
        ['Poor', 'Moderate', 'Good']
    ]

    # --- Models for Survival Rate and Cost (Regressors) ---
    # Filter data for cancer cases only
    cancer_cases = liver_data[liver_data['Prediction'] == 'Yes'].copy()

    if cancer_cases.empty:
        print("\n⚠️ No 'Yes' predictions found in the dataset for regression models.")
        return

    # Survival Rate Regressor
    features_surv = cancer_cases.drop(columns=['Prediction', 'Survival_Rate'], errors='ignore')
    numerical_cols_surv = ['Age', 'Incidence_Rate', 'Mortality_Rate', 'Cost_of_Treatment']
    y_survival = cancer_cases['Survival_Rate']
    
    X_train_surv, X_test_surv, y_surv_train, y_surv_test = train_test_split(
        features_surv, y_survival, test_size=0.25, random_state=42
    )

    ohe_surv = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ord_enc_surv = OrdinalEncoder(categories=ordinal_categories)
    scaler_surv = StandardScaler()
    label_encoders_surv = {col: LabelEncoder() for col in label_cols}

    X_train_surv_final = pd.concat([
        pd.DataFrame(ohe_surv.fit_transform(X_train_surv[onehot_cols]), columns=ohe_surv.get_feature_names_out(onehot_cols), index=X_train_surv.index),
        pd.DataFrame(ord_enc_surv.fit_transform(X_train_surv[ordinal_cols]), columns=ordinal_cols, index=X_train_surv.index),
        X_train_surv[label_cols].apply(lambda col: label_encoders_surv[col.name].fit_transform(col)),
        pd.DataFrame(scaler_surv.fit_transform(X_train_surv[numerical_cols_surv]), columns=numerical_cols_surv, index=X_train_surv.index)
    ], axis=1)

    X_test_surv_final = pd.concat([
        pd.DataFrame(ohe_surv.transform(X_test_surv[onehot_cols]), columns=ohe_surv.get_feature_names_out(onehot_cols), index=X_test_surv.index),
        pd.DataFrame(ord_enc_surv.transform(X_test_surv[ordinal_cols]), columns=ordinal_cols, index=X_test_surv.index),
        X_test_surv[label_cols].apply(lambda col: label_encoders_surv[col.name].transform(col)),
        pd.DataFrame(scaler_surv.transform(X_test_surv[numerical_cols_surv]), columns=numerical_cols_surv, index=X_test_surv.index)
    ], axis=1)
    
    survival_regressor = XGBRegressor(random_state=42)
    survival_regressor.fit(X_train_surv_final, y_surv_train)
    y_surv_preds = survival_regressor.predict(X_test_surv_final)
    print("✅ Survival Rate Model Trained!")
    print("\n🔍 Survival Rate Model Performance:")
    print(f"R2 Score: {r2_score(y_surv_test, y_surv_preds):.4f}")
    print(f"MAE: {mean_absolute_error(y_surv_test, y_surv_preds):.4f}")

    # Cost Regressor
    features_cost = cancer_cases.drop(columns=['Prediction', 'Cost_of_Treatment'], errors='ignore')
    numerical_cols_cost = ['Age', 'Incidence_Rate', 'Mortality_Rate', 'Survival_Rate']
    y_cost = cancer_cases['Cost_of_Treatment']
    
    X_train_cost, X_test_cost, y_cost_train, y_cost_test = train_test_split(
        features_cost, y_cost, test_size=0.25, random_state=42
    )

    ohe_cost = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ord_enc_cost = OrdinalEncoder(categories=ordinal_categories)
    scaler_cost = StandardScaler()
    label_encoders_cost = {col: LabelEncoder() for col in label_cols}

    X_train_cost_final = pd.concat([
        pd.DataFrame(ohe_cost.fit_transform(X_train_cost[onehot_cols]), columns=ohe_cost.get_feature_names_out(onehot_cols), index=X_train_cost.index),
        pd.DataFrame(ord_enc_cost.fit_transform(X_train_cost[ordinal_cols]), columns=ordinal_cols, index=X_train_cost.index),
        X_train_cost[label_cols].apply(lambda col: label_encoders_cost[col.name].fit_transform(col)),
        pd.DataFrame(scaler_cost.fit_transform(X_train_cost[numerical_cols_cost]), columns=numerical_cols_cost, index=X_train_cost.index)
    ], axis=1)
    
    X_test_cost_final = pd.concat([
        pd.DataFrame(ohe_cost.transform(X_test_cost[onehot_cols]), columns=ohe_cost.get_feature_names_out(onehot_cols), index=X_test_cost.index),
        pd.DataFrame(ord_enc_cost.transform(X_test_cost[ordinal_cols]), columns=ordinal_cols, index=X_test_cost.index),
        X_test_cost[label_cols].apply(lambda col: label_encoders_cost[col.name].transform(col)),
        pd.DataFrame(scaler_cost.transform(X_test_cost[numerical_cols_cost]), columns=numerical_cols_cost, index=X_test_cost.index)
    ], axis=1)

    cost_regressor = XGBRegressor(random_state=42)
    cost_regressor.fit(X_train_cost_final, y_cost_train)
    y_cost_preds = cost_regressor.predict(X_test_cost_final)
    print("\n✅ Treatment Cost Model Trained!")
    print("\n🔍 Treatment Cost Model Performance:")
    print(f"R2 Score: {r2_score(y_cost_test, y_cost_preds):.4f}")
    print(f"MAE: {mean_absolute_error(y_cost_test, y_cost_preds):.4f}")
    
    # Save all models and preprocessing objects
    joblib.dump(survival_regressor, 'survival_regressor.joblib')
    joblib.dump(cost_regressor, 'cost_regressor.joblib')

    joblib.dump(ohe_surv, 'ohe_surv.joblib')
    joblib.dump(ord_enc_surv, 'ord_enc_surv.joblib')
    joblib.dump(scaler_surv, 'scaler_surv.joblib')
    joblib.dump(label_encoders_surv, 'label_encoders_surv.joblib')

    joblib.dump(ohe_cost, 'ohe_cost.joblib')
    joblib.dump(ord_enc_cost, 'ord_enc_cost.joblib')
    joblib.dump(scaler_cost, 'scaler_cost.joblib')
    joblib.dump(label_encoders_cost, 'label_encoders_cost.joblib')

    print("\n💾 All models and preprocessing objects saved successfully!")

if __name__ == "__main__":
    data_file_path = "C:\\Datasets\\liver_cancer_prediction.csv"
    train_and_save_regressors(data_file_path)