# Importing necessary libraries
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import json

app = Flask(__name__)

# Sample data for career prediction
data = pd.read_csv('career_data.csv')  # Update with the correct path to your dataset

# Data preprocessing
# Convert the 'skills' column to string and split it
data['skills'] = data['skills'].apply(lambda x: list(map(int, str(x).split(','))) if isinstance(x, str) else [x])
data['interest'] = data['interest'].astype('category')

# One-hot encoding for the 'interest' column
data = pd.get_dummies(data, columns=['interest'], drop_first=True)

# Prepare features and labels
X = data[['aptitude_score', 'marks_10th', 'marks_12th'] + list(data.columns[data.columns.str.startswith('interest_')])]
y = data['domain']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting data from the incoming request
        user_data = request.json

        # Extract user inputs
        aptitude_score = user_data['aptitude_score']
        current_skills = user_data['current_skills']
        interests_goals = user_data['interests_goals']
        math_marks = user_data['math_marks']
        science_marks = user_data['science_marks']

        # Prepare input data
        input_data = pd.DataFrame([{
            'aptitude_score': aptitude_score,
            'marks_10th': math_marks,
            'marks_12th': science_marks
        }])

        # Create a binary representation for current skills
        skills_df = pd.DataFrame(0, index=input_data.index, columns=range(1, 100))  # Assuming skills range from 1 to 99
        for skill in current_skills:
            skills_df.at[0, skill] = 1  # Marking skills present

        # Combine input data with skills
        input_data_encoded = pd.concat([input_data, skills_df], axis=1)

        # One-hot encoding for the 'interests_goals' column
        interest_encoded = pd.get_dummies(pd.DataFrame({'interest': [interests_goals]}), drop_first=True)
        input_data_encoded = pd.concat([input_data_encoded, interest_encoded], axis=1)

        # Align the input data with the training data
        input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)

        # Perform the prediction using the Random Forest model
        prediction = rf_model.predict(input_data_encoded)[0]

        # Return the prediction as a response
        return jsonify({'predicted_domain': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

# Running the Flask application
if __name__ == '__main__':
    app.run(debug=True)
