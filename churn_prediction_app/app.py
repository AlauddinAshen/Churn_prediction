
import streamlit as st
import pandas as pd
import joblib

model=joblib.load('churn_prediction_model.pkl')
columns=joblib.load('columns.pkl')

st.title('Customer Churn Prediction')
st.write("Enter customer details to predict churn ")
gender = st.selectbox('Gender', ['Female', 'Male'])
senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
partner = st.selectbox('Partner', ['No', 'Yes'])
dependents = st.selectbox('Dependents', ['No', 'Yes'])
tenure = st.slider('Tenure (months)', 0, 72)
monthly=st.number_input('Monthly Charges', min_value=0.0)
total=st.number_input('Total Charges', min_value=0.0)
phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
multiple_lines = st.selectbox('Multiple Lines', ['No phone service', 'No', 'Yes'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])

input_data = {
  'gender': gender,
  'SeniorCitizen': senior_citizen,
  'partner': partner,
  'dependents': dependents,
  'tenure': tenure,
  'MonthlyCharges': monthly,
  'TotalCharges': total,
  'PhoneService': phone_service,
  'MultipleLines': multiple_lines,
  'InternetService': internet_service
}

def predict_chrun(data):
  df_input=pd.DataFrame([data])
  df_encoded=pd.get_dummies(df_input).reindex(columns=columns,fill_value=0)
  prediction=model.predict(df_encoded)
  return "Churn" if prediction[0]==1 else "No Churn"

if st.button('Predict'):
  result=predict_chrun(input_data)
  st.write(f'Prediction: {result}')
