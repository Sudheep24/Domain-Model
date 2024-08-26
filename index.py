from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Initialize FastAPI app
app = FastAPI()

# Define input data schema
class InputData(BaseModel):
    currentSkills: list[int]
    aptitudeScore: int
    mathMarks: int
    scienceMarks: int
    interestsGoals: int

# Load models and scaler
def load_models():
    nn_model = load_model('model.h5')
    gbm_model = joblib.load('gbm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return nn_model, gbm_model, scaler, label_encoder

# Load models on startup
nn_model, gbm_model, scaler, label_encoder = load_models()

# Prediction endpoint
@app.post("/predict")
def predict(input_data: InputData):
    # Convert input data to DataFrame format
    user_data = {
        'current_skills': np.mean(input_data.currentSkills),  # Average the skills if multiple are provided
        'aptitude_score': input_data.aptitudeScore,
        'math_marks': input_data.mathMarks,
        'science_marks': input_data.scienceMarks,
        'interests_goals': input_data.interestsGoals
    }
    
    user_data_df = pd.DataFrame([user_data])
    user_data_scaled = scaler.transform(user_data_df)
    
    # Predict using the neural network
    nn_predictions = nn_model.predict(user_data_scaled)
    nn_prediction_class = np.argmax(nn_predictions, axis=1)  # Convert to class label
    
    # Use NN prediction as input for Gradient Boosting model
    gbm_prediction = gbm_model.predict(nn_prediction_class.reshape(-1, 1))
    gbm_domain_index = gbm_prediction[0]
    
    # Map index back to domain
    gbm_domain = label_encoder.inverse_transform([gbm_domain_index])[0]
    
    return {"recommended_domain": gbm_domain}
