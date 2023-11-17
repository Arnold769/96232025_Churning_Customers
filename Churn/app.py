import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler

# Load the Keras model
model = load_model('/Users/arnoldaryeequaye/Desktop/Churn/saved_model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit UI
st.title('Customer Churn Predictor')

# Assuming 'tenure', 'MonthlyCharges', and 'TotalCharges' as input features
tenure = st.number_input('Enter Tenure')
monthly_charges = st.number_input('Enter Monthly Charges')
total_charges = st.number_input('Enter Total Charges')

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two years"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges_sidebar = st.sidebar.number_input("Monthly Charges", min_value=1, max_value=1000, value=50)
total_charges_sidebar = st.sidebar.number_input("Total Charges", min_value=0, max_value=100000, value=0)

# Create a list with the user input
my_list = [total_charges, monthly_charges, tenure]
my_list = np.array(my_list).reshape(1, -1)

# Create a DataFrame with the reshaped user input
user_input = pd.DataFrame(my_list, columns=["TotalCharges", "MonthlyCharges", "tenure"])

# Scale the user input using the loaded scaler
scaler = StandardScaler()

# Make predictions
prediction = model.predict(scaler.fit_transform(user_input))

# Display prediction
st.write(f'Prediction: {"Churn" if prediction[0] >= 0.5 else "No Churn"}')
