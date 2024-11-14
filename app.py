import streamlit as st
import joblib  # for loading the scaler
import numpy as np
import pickle  # for loading the model

# Load the scaler with joblib
  # Ensure you're loading the correct file
scaler = joblib.load('scaler.joblib')
  
# Load the model (saved with pickle)
with open('xgboost_model (1).pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Energy Production Predictor')
st.subheader('Enter Input Parameters:')

# Input fields for parameters
temperature = st.number_input('Temperature', min_value=-20.0, max_value=40.0, value=30.0)
exhaust_vacuum = st.number_input('Exhaust Vacuum', min_value=25.0, max_value=85.0, value=70.0)
amb_pressure = st.number_input('Ambient Pressure', min_value=985.0, max_value=1035.0, value=1000.0)
r_humidity = st.number_input('Relative Humidity', min_value=20.0, max_value=100.0, value=50.0)

if st.button('Predict Energy Production'):
    try:
        # Create input array
        input_data = np.array([[temperature, exhaust_vacuum, amb_pressure, r_humidity]])

        # Scale input data using the loaded scaler
        scaled_data = scaler.transform(input_data)

        # Make prediction using the model
        prediction = model.predict(scaled_data)

        # Display the prediction
        st.success(f'Predicted Energy Production: {prediction[0]:.2f} MW')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')

# Sidebar with additional info
st.sidebar.markdown('''
### About
This app predicts energy production based on environmental parameters using an XGBoost model.
''')
