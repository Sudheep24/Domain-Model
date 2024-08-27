from flask import Flask, request, jsonify
import joblib
import numpy as np
import tensorflow as tf
from typing import List

# Create Flask app
app = Flask(__name__)

# Load the trained model and components
def load_models():
    model = tf.keras.models.load_model('model.h5')
    label_encoder = joblib.load('label_encoder.pkl')
    mlb = joblib.load('mlb.pkl')
    return model, label_encoder, mlb

model, label_encoder, mlb = load_models()

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_recommendation():
    # Get the request data
    data = request.json
    user_skills = data.get('currentSkills', [])
    aptitude_score = data.get('aptitudeScore', 0)
    math_marks = data.get('mathMarks', 0)
    science_marks = data.get('scienceMarks', 0)
    interests_goals = data.get('interestsGoals', 0)

    # Validate input data
    if not isinstance(user_skills, list) or not isinstance(aptitude_score, int) or not isinstance(math_marks, int) or not isinstance(science_marks, int) or not isinstance(interests_goals, int):
        return jsonify({'error': 'Invalid input'}), 400

    # Preprocess user input
    skills_binarized = mlb.transform([user_skills])
    user_input_array = np.hstack([
        skills_binarized,
        np.array([[aptitude_score, math_marks, science_marks, interests_goals]])
    ])

    # Ensure the input shape matches the model's expected input shape
    if user_input_array.shape[1] != model.input_shape[1]:
        return jsonify({'error': 'Input shape mismatch'}), 400

    # Make a prediction
    nn_predictions = model.predict(user_input_array)
    predicted_domain_idx = np.argmax(nn_predictions, axis=1)
    predicted_domain = label_encoder.inverse_transform(predicted_domain_idx)

    return jsonify({"recommended_domain": predicted_domain[0]})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

