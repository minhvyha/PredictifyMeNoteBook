from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Iris model API'}
@app.get('/get')
def get():
    return {'message': 'Iris model API'}



@app.post('/predict')
def predict(data: dict):
    with open('app/heartattack/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    with open('app/heartattack/heartattack.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('app/heartattack/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)


        # Extract features from input data
    input_data = pd.DataFrame([data])

    # Apply pd.get_dummies for one-hot encoding
    encoded_data = pd.get_dummies(
        input_data,
        columns=['chest_pain_type', 'resting_ecg', 'st_slope'],
        drop_first=False
    )

    # Add missing columns and reorder
    for col in feature_names:
        if col not in encoded_data:
            encoded_data[col] = 0  # Add missing columns as zeros
    encoded_data = encoded_data[feature_names]  # Reorder to match training

    # Scale features
    features = scaler.transform(encoded_data)

    # Make predictions
    prediction = model.predict(features)

    return {'features': features.tolist(), 'prediction': prediction.tolist()}


