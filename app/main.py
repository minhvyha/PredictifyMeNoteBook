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

# Load pre-trained artifacts once to avoid reloading for every request
with open('app/heartattack/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('app/heartattack/heartattack.pkl', 'rb') as f:
    model = pickle.load(f)

with open('app/heartattack/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)



@app.post('/predict')
def predict(data: dict):
    # Validate input
    required_keys = [
        'age', 'sex', 'resting_bp_s', 'cholesterol', 'fasting_blood_sugar',
        'max_heart_rate', 'exercise_angina', 'oldpeak', 'chest_pain_type',
        'resting_ecg', 'st_slope'
    ]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise HTTPException(status_code=400, detail=f"Missing keys in input: {', '.join(missing_keys)}")

    # Convert input to DataFrame
    input_data = pd.DataFrame([data])

    # One-hot encode categorical columns
    encoded_data = pd.get_dummies(
        input_data,
        columns=['chest_pain_type', 'resting_ecg', 'st_slope'],
        drop_first=False
    )

    # Reindex to match training columns
    encoded_data = encoded_data.reindex(columns=feature_names, fill_value=0)

    # Scale features
    features = scaler.transform(encoded_data)

    # Make predictions
    prediction = model.predict(features)

    return {'features': features.tolist(), 'prediction': prediction.tolist()}

