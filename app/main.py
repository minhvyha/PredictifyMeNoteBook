from fastapi import FastAPI
import joblib
import numpy as np
import pickle
print("NumPy version:", np.__version__)

# model = joblib.load('app/model.joblib')

with open('app/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('app/heartattack.pkl', 'rb') as f:
    model = pickle.load(f)


app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Iris model API'}
@app.get('/get')
def get():
    return {'message': 'Iris model API'}



@app.post('/predict')
def predict(data: dict):
    required_fields = [
        'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 
        'fasting_blood_sugar', 'resting_ecg', 'max_heart_rate', 
        'exercise_angina', 'oldpeak', 'st_slope'
    ]
    

        # Extract features from input data
    features = np.array([
        data['age'],
        data['sex'],
        data['resting_bp_s'],
        data['cholesterol'],
        data['fasting_blood_sugar'],
        data['max_heart_rate'],
        data['exercise_angina'],
        data['oldpeak'],
        data['chest_pain_type_1'],
        data['chest_pain_type_2'],
        data['chest_pain_type_3'],
        data['chest_pain_type_4'],
        data['resting_ecg_0'],
        data['resting_ecg_1'],
        data['resting_ecg_2'],
        data['st_slope_0'],
        data['st_slope_1'],
        data['st_slope_2'],
        data['st_slope_3']
    ])
    features = features.reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)


    return {'features': features.tolist(), 'prediction': prediction.tolist()}