from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

app = FastAPI()

# Load models and scaler
nn_model = load_model('model.h5')
gbm_model = joblib.load('gbm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

class UserData(BaseModel):
    current_skills: int
    aptitude_score: int
    math_marks: int
    science_marks: int
    interests_goals: int

@app.post("/predict/")
def predict(user_data: UserData):
    data = pd.DataFrame([user_data.dict()])
    data_scaled = scaler.transform(data)
    
    nn_predictions = nn_model.predict(data_scaled)
    nn_prediction_class = np.argmax(nn_predictions, axis=1)
    
    gbm_prediction = gbm_model.predict(nn_prediction_class.reshape(-1, 1))
    gbm_domain_index = gbm_prediction[0]
    
    gbm_domain = label_encoder.inverse_transform([gbm_domain_index])[0]
    
    return {"recommended_domain": gbm_domain}
