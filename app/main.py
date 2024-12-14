from fastapi import FastAPI
import numpy as np
import pickle

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

    new_data = [
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

    features = np.array(new_data)
    features = features.reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)


    return {'features': features.tolist(), 'prediction': prediction.tolist()}