from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib

app = Flask(__name__)

# Load models
nn_model = load_model('model.h5')
gbm_model = joblib.load('gbm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_data = [data['current_skills'], data['aptitude_score'], data['math_marks'], data['science_marks'], data['interests_goals']]
    user_data_df = pd.DataFrame([user_data], columns=['current_skills', 'aptitude_score', 'math_marks', 'science_marks', 'interests_goals'])
    user_data_scaled = scaler.transform(user_data_df)
    
    nn_predictions = nn_model.predict(user_data_scaled)
    nn_prediction_class = np.argmax(nn_predictions, axis=1)
    
    gbm_prediction = gbm_model.predict(nn_prediction_class.reshape(-1, 1))
    gbm_domain_index = gbm_prediction[0]
    gbm_domain = label_encoder.inverse_transform([gbm_domain_index])[0]
    
    return jsonify({'recommended_domain': gbm_domain})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
