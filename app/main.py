from fastapi import FastAPI
import joblib
import numpy as np
import pickle

# model = joblib.load('app/model.joblib')

with open('app/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('app/heartattack.pkl', 'rb') as f:
    scaler = pickle.load(f)


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
    
    # Check if all required fields are present
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required fields: {', '.join(missing_fields)}"
        )

    try:
        # Extract features from input data
        features = np.array([
            data['age'],
            data['sex'],
            data['chest_pain_type'],
            data['resting_bp'],
            data['cholesterol'],
            data['fasting_blood_sugar'],
            data['resting_ecg'],
            data['max_heart_rate'],
            data['exercise_angina'],
            data['oldpeak'],
            data['st_slope']
        ])

        # Reshape features to 2D array (one sample)
        features = features.reshape(1, -1)

        # Apply the same scaler used during training
        features_scaled = scaler.transform(features)

        # Predict using the trained model
        prediction = model.predict(features_scaled)

        # Return the prediction result
        return {'prediction': prediction[0]}
    
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Key error: {str(e)}")