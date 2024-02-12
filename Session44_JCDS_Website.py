import streamlit as sl
import pickle
import numpy as np
import pandas as pd

# write the title
sl.title("HALOO")

# load the preprocess
if 'impute' not in sl.session_state:
    sl.session_state['impute'] = pickle.load(open('Imputation.sav', 'rb'))

if 'encoder' not in sl.session_state:
    sl.session_state['encoder'] = pickle.load(open('Feature Encoding.sav', 'rb'))

# load the model
if 'model' not in sl.session_state:
    sl.session_state['model'] = pickle.load(open('Model LR Smote.sav', 'rb'))

"""
    Selamat datang di Website kami!
    \nWebsite ini dapat membantu anda untuk memperoleh informasi lebih cepat terkait stroke
"""

gender_input = sl.selectbox(
    "Insert your Gender :",
    ('Male', 'Female')
    )
age_input = sl.number_input("Enter your age:")
hypertension_input = sl.number_input("Enter your hypertension rate:")
heart_rate_input = sl.number_input("Enter your heart rate:")
is_married_input = sl.selectbox(
    "Have you ever married?",
    ('Yes', 'No')
    )
work_type_input = sl.selectbox(
    "What is your work type?",
    ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked')
    )
residence_type_input = sl.selectbox(
    "What is your residence type?",
    ('Urban', 'Rural')
    )
avg_glucose_input = sl.number_input("Enter your average glucose:")
bmi_input = sl.number_input("Enter your BMI:")

is_smoke_input = sl.selectbox(
    "Do you smoke?",
    ('formerly smoked', 'never smoked', 'smokes', 'Unknown')
    )


if sl.button("Dapatkan Prediksi"):
    feature_list = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status']
    
    data = np.array([gender_input, age_input, hypertension_input, heart_rate_input, is_married_input, work_type_input, residence_type_input, avg_glucose_input, bmi_input, is_smoke_input]).reshape(1,-1)
    data_df = pd.DataFrame(data, columns = feature_list)

    data_imputed = pd.DataFrame(
        sl.session_state['impute'].transform(data_df),
        columns = ['gender', 'bmi', 'age', 'hypertension', 'heart_disease', 'ever_married',
                    'work_type', 'Residence_type', 'avg_glucose_level', 'smoking_status'])
    data_encoded = sl.session_state['encoder'].transform(data_imputed)

    # sl.write(data_encoded)
    prediction = sl.session_state['model'].predict(data_encoded)
    diabetes_prediction = "Tidak berisiko stroke"
    if prediction[0] == 1:
        diabetes_prediction = "Berisiko Stroke!"
    sl.write(f'Prediksi: {diabetes_prediction}')

    """
        Nb:
        \nIni hanya sebuah prediksi. Harap hubungi dokter untuk memperoleh penanganan lebih detail
    """
else:
    sl.write("Please input the feature above to start modelling")