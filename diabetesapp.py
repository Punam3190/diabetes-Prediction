# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:20:48 2023

@author: Punam
"""

# Import necessary libraries
import numpy as np
import pickle
import streamlit as st

# Load the saved model
model_file_path = "C:/Users/Punam/Downloads/trained_model.sav"
loaded_model = pickle.load(open(model_file_path, 'rb'))

# Create a function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

def main():
    # Set a title for your app
    st.title('Diabetes Prediction App')

    # Create input fields for user data
    input_data = {}
    input_data['Pregnancies'] = st.number_input('Number of Pregnancies', min_value=0, max_value=100, value=0)
    input_data['Glucose'] = st.number_input('Glucose Level', min_value=0, value=0)
    input_data['BloodPressure'] = st.number_input('Blood Pressure value', min_value=0, value=0)
    input_data['SkinThickness'] = st.number_input('Skin Thickness value', min_value=0, value=0)
    input_data['Insulin'] = st.number_input('Insulin Level', min_value=0, value=0)
    input_data['BMI'] = st.number_input('BMI value', min_value=0, value=0)
    input_data['DiabetesPedigreeFunction'] = st.number_input('Diabetes Pedigree Function value', min_value=0.0, value=0.0)
    input_data['Age'] = st.number_input('Age of the Person', min_value=0, max_value=120, value=30)

    # Perform prediction when the user clicks the button
    if st.button('Diabetes Test Result'):
        result = diabetes_prediction(list(input_data.values()))
        if result == 0:
            diagnosis = 'The person is not diabetic'
        else:
            diagnosis = 'The person is diabetic'

        st.success(diagnosis)

if __name__ == '__main__':
    main()
