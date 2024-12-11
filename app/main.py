from fastapi import FastAPI
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
    """
    Predict the likelihood of a heart attack based on input data.

    Input:
    - data: JSON object with numeric features.
    
    Example input:
    {
        "numeric_features": [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    }
    """
    try:
        # Extract numeric features
        numeric_data = np.array(data['numeric_features']).reshape(1, -1)
        
        # Scale the features
        scaled_data = scaler.transform(numeric_data)
        
        # Make a prediction
        prediction = model.predict(scaled_data)

        # Return the result
        return {'heart_attack_risk': prediction[0]}
    except KeyError as e:
        return {'error': f'Missing key in input data: {str(e)}'}
    except Exception as e:
        return {'error': str(e)}