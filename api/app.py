from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

app = Flask(__name__)

# Load models and other resources
nn_model = load_model('model.h5')
gbm_model = joblib.load('gbm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Ensure the input data contains all required fields
    required_fields = ['current_skills', 'aptitude_score', 'math_marks', 'science_marks', 'interests_goals']
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Invalid input data"}), 400

    user_data = [
        data['current_skills'],
        data['aptitude_score'],
        data['math_marks'],
        data['science_marks'],
        data['interests_goals']
    ]

    # Convert to DataFrame and scale the input
    user_data_df = pd.DataFrame([user_data], columns=required_fields)
    user_data_scaled = scaler.transform(user_data_df)

    # Predict with Neural Network
    nn_predictions = nn_model.predict(user_data_scaled)
    nn_prediction_class = np.argmax(nn_predictions, axis=1)  # Convert to class label

    # Predict with Gradient Boosting Model
    gbm_prediction = gbm_model.predict(nn_prediction_class.reshape(-1, 1))
    gbm_domain_index = gbm_prediction[0]

    # Decode the predicted domain
    gbm_domain = label_encoder.inverse_transform([gbm_domain_index])[0]

    return jsonify({"recommended_domain": gbm_domain})

if __name__ == '__main__':
    app.run(debug=True)
