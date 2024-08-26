from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

app = Flask(__name__)

# Load models and other objects
def load_models():
    nn_model = load_model('model.h5')
    gbm_model = joblib.load('gbm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return nn_model, gbm_model, scaler, label_encoder

nn_model, gbm_model, scaler, label_encoder = load_models()

def predict_recommendation(defaultValues):
    user_data = {
        'current_skills': np.mean([int(skill) for skill in defaultValues['currentSkills']]),
        'aptitude_score': int(defaultValues['aptitudeScore']),
        'math_marks': int(defaultValues['mathMarks']),
        'science_marks': int(defaultValues['scienceMarks']),
        'interests_goals': int(defaultValues['interestsGoals'])
    }
    
    user_data_df = pd.DataFrame([user_data])
    user_data_scaled = scaler.transform(user_data_df)
    
    nn_predictions = nn_model.predict(user_data_scaled)
    nn_prediction_class = np.argmax(nn_predictions, axis=1)
    
    gbm_prediction = gbm_model.predict(nn_prediction_class.reshape(-1, 1))
    gbm_domain_index = gbm_prediction[0]
    gbm_domain = label_encoder.inverse_transform([gbm_domain_index])[0]
    
    return gbm_domain

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = request.json
        recommended_domain = predict_recommendation(input_data)
        return jsonify({'recommended_domain': recommended_domain})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
