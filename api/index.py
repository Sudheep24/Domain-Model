import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and Prepare Data
def load_and_prepare_data():
    df = pd.DataFrame({
        'current_skills': [4, 5, 3, 2, 4, 5, 3, 4, 2, 5],
        'aptitude_score': [85, 90, 75, 80, 88, 92, 78, 80, 70, 88],
        'math_marks': [90, 92, 85, 78, 88, 91, 80, 82, 72, 89],
        'science_marks': [92, 90, 80, 75, 85, 94, 78, 81, 70, 90],
        'interests_goals': [3, 2, 4, 1, 5, 4, 3, 2, 4, 5],
        'domain': [
            'Full Stack Software Development', 'App Development', 'Data Science',
            'Machine Learning', 'Cybersecurity', 'Robotics', 'Automotive Engineering',
            'Thermodynamics', 'Structural Analysis', 'Geotechnical Engineering'
        ]
    })
    
    X = df[['current_skills', 'aptitude_score', 'math_marks', 'science_marks', 'interests_goals']]
    y = df['domain']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    return X, y_encoded, le

# Neural Network Model
def build_nn_model(input_dim, num_categories):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_categories, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Gradient Boosting Model
def build_gbm_model():
    return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train and Evaluate Models
def train_evaluate_models(X, y, label_encoder):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine number of unique classes
    num_classes = len(label_encoder.classes_)
    print(f"Total number of unique classes: {num_classes}")

    # Check class distribution
    from collections import Counter
    print("Class distribution:", Counter(y))

    # Define and train the Neural Network model
    nn_model = build_nn_model(X_scaled.shape[1], num_classes)
    nn_model.fit(X_scaled, y, epochs=20, batch_size=2, verbose=1, validation_split=0.2)

    # Get NN predictions to use as features for GBM
    nn_predictions = nn_model.predict(X_scaled)
    nn_predictions_class = np.argmax(nn_predictions, axis=1)  # Convert to class labels
    
    # Define and train the Gradient Boosting model
    gbm_model = build_gbm_model()
    gbm_model.fit(nn_predictions_class.reshape(-1, 1), y)  # Train GBM with NN predictions
    
    # Save the models and scaler
    nn_model.save('model.h5')
    joblib.dump(gbm_model, 'gbm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Models, scaler, and label encoder saved.")

    return nn_model, gbm_model, scaler

# Load saved models
def load_models():
    nn_model = load_model('model.h5')
    gbm_model = joblib.load('gbm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return nn_model, gbm_model, scaler, label_encoder

# Check if models exist, if not train and save them
def ensure_models_exist(X, y, label_encoder):
    try:
        # Attempt to load models
        nn_model, gbm_model, scaler, le = load_models()
        print("Loaded existing models and scaler.")
    except FileNotFoundError:
        # If models do not exist, train and save them
        print("Models not found. Training new models...")
        nn_model, gbm_model, scaler = train_evaluate_models(X, y, label_encoder)
        le = label_encoder
    return nn_model, gbm_model, scaler, le

# Predict and Recommend Domains using Neural Network Output for GBM
def predict_recommendation(user_data, nn_model, gbm_model, scaler, label_encoder):
    user_data_df = pd.DataFrame([user_data], columns=['current_skills', 'aptitude_score', 'math_marks', 'science_marks', 'interests_goals'])
    user_data_scaled = scaler.transform(user_data_df)
    
    nn_predictions = nn_model.predict(user_data_scaled)
    nn_prediction_class = np.argmax(nn_predictions, axis=1)  # Convert to class label
    
    # Use NN prediction as input for GBM
    gbm_prediction = gbm_model.predict(nn_prediction_class.reshape(-1, 1))
    gbm_domain_index = gbm_prediction[0]
    
    gbm_domain = label_encoder.inverse_transform([gbm_domain_index])[0]
    
    print(f"Recommended Domain (Gradient Boosting): {gbm_domain}")

# Main Execution
X, y, label_encoder = load_and_prepare_data()
nn_model, gbm_model, scaler, le = ensure_models_exist(X, y, label_encoder)

# Sample User Data for Prediction
sample_user_data = [4, 80, 90, 92, 3]
predict_recommendation(sample_user_data, nn_model, gbm_model, scaler, le)

# Example of adding new data (if needed)
def add_new_data(new_data, existing_data, existing_labels, label_encoder):
    new_df = pd.DataFrame(new_data)
    updated_data = pd.concat([existing_data, new_df], ignore_index=True)
    # Generate labels for the new data
    new_labels = [existing_labels[0]] * len(new_df)  # Use default or specify
    updated_labels = np.concatenate([existing_labels, new_labels])
    return updated_data, updated_labels

# Example of retraining models with new data
def retrain_models(existing_data, existing_labels):
    X_train, X_test, y_train, y_test = train_test_split(existing_data, existing_labels, test_size=0.2, random_state=42)
    nn_model, gbm_model, scaler = train_evaluate_models(X_train, y_train, label_encoder)
    return nn_model, gbm_model, scaler