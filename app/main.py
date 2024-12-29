from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import pandas as pd

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://predictifyme.vercel.app"],  # Update this with the specific origin(s) you want to allow
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get('/')
def read_root():
    return {'message': 'Iris model API'}
@app.get('/get')
def get():
    return {'message': 'Iris model API'}

# Load pre-trained artifacts once to avoid reloading for every request
with open('app/heartattack/scaler.pkl', 'rb') as f:
    heartattack_scaler = pickle.load(f)

with open('app/heartattack/heartattack.pkl', 'rb') as f:
    heartattack_model = pickle.load(f)

with open('app/diabetes/diabetes.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

with open('app/diabetes/scaler.pkl', 'rb') as f:
    diabetes_scaler = pickle.load(f)


@app.post('/predict')
def predict(data: dict):

    heartattack_data = [
        data["age"],  # Adjusted
        data["sex"],  # Adjusted
        data["resting_bp_s"],  # Adjusted
        data["cholesterol"],  # Adjusted
        data["fasting_blood_sugar"],  # Adjusted
        data["max_heart_rate"],  # Adjusted
        data["exercise_angina"],  # Adjusted
        data["oldpeak"],  # Adjusted

        # Chest pain type (one-hot encoded)
        1 if data["chest_pain_type"] == 1 else 0,
        1 if data["chest_pain_type"] == 2 else 0,  # Corresponding to the original chest_pain_type value of 2
        1 if data["chest_pain_type"] == 3 else 0,
        1 if data["chest_pain_type"] == 4 else 0,

        # Resting ECG (one-hot encoded)
        1 if data["resting_ecg"] == 0 else 0,
        1 if data["resting_ecg"] == 1 else 0,
        1 if data["resting_ecg"] == 2 else 0,

        # ST slope (one-hot encoded)
        1 if data["st_slope"] == 0 else 0,
        1 if data["st_slope"] == 1 else 0,
        1 if data["st_slope"] == 2 else 0,
        1 if data["st_slope"] == 3 else 0,
    ]
    
    # Exclude data['sex'] for initial diabetes_data list
    diabetes_data = [
        data["age"],
        data["hypertension"],
        data["heart_disease"],
        data["bmi"],
        data["HbA1c_level"],
        data["blood_glucose_level"]
    ]

    df = pd.DataFrame([data])
    
    # Example preprocessing: One-hot encode categorical columns
    categorical_columns = ['chest_pain_type', 'resting_ecg', 'st_slope']
    df = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns)
    df = np.array(df)

    # Convert the diabetes_data into a NumPy array and reshape it
    diabetes_features = np.array(diabetes_data).reshape(1, -1)
    # Scale the features
    diabetes_features = diabetes_scaler.transform(diabetes_features)

    # Add 'sex' after scaling
    diabetes_features = np.insert(diabetes_features, 0, data["sex"], axis=1)
    diabetes_prediction = diabetes_model.predict(diabetes_features)

    heartattack_features = np.array(heartattack_data)
    heartattack_features = heartattack_features.reshape(1, -1)
    heartattack_features = heartattack_scaler.transform(heartattack_features)
    heartattack_prediction = heartattack_model.predict(heartattack_features)


    return {'heartattack': heartattack_prediction.tolist(), 'diabetes': diabetes_prediction.tolist()}