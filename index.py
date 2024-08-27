import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import joblib
import tensorflow as tf

# Mapping of domains to skill IDs
domain_to_skills = {
    "Full Stack Software Development": [1, 2, 3, 4, 6, 7, 11, 14, 16],
    "Data Science": [19, 20, 21, 22, 23, 24, 25, 27, 28],
    "Machine Learning Engineering": [32, 35, 36, 37, 38, 39, 40, 42, 43],
    "Cybersecurity": [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
    "Software Engineering": [57, 58, 59, 60, 62, 63, 65, 68, 69],
    "Cloud Engineering": [70, 71, 72, 73, 75, 77, 79, 82, 84],
    "Web Development": [85, 86, 87, 88, 89, 91, 92, 94, 96],
    "DevOps": [98, 99, 101, 102, 103, 105, 106, 107, 108],
    "Game Development": [110, 111, 112, 113, 114, 116, 118],
    "IT Support": [120, 121, 123, 126, 128, 130, 131]
}

# Mapping of skill IDs to skill names
skill_id_to_name = {
    "1": "HTML", "2": "CSS", "3": "JavaScript", "4": "React.js", "5": "Angular.js", "6": "Node.js", "7": "Express.js",
    "8": "Python", "9": "Ruby on Rails", "10": "Java", "11": "SQL", "12": "NoSQL", "13": "MongoDB", "14": "Git",
    "15": "GitHub", "16": "Docker", "17": "Kubernetes", "18": "CI/CD pipelines", "19": "R", "20": "Pandas",
    "21": "NumPy", "22": "Scikit-Learn", "23": "TensorFlow", "24": "Keras", "25": "Matplotlib", "26": "Seaborn",
    "27": "Tableau", "28": "Hadoop", "29": "Spark", "30": "PyTorch", "31": "Flask", "32": "Regression",
    "33": "Classification", "34": "Clustering", "35": "Firewalls", "36": "VPNs", "37": "IDS/IPS",
    "38": "Encryption algorithms", "39": "PKI", "40": "Penetration testing", "41": "Risk analysis",
    "42": "Forensics", "43": "Log analysis", "44": "Wireshark", "45": "Metasploit", "46": "Nessus",
    "47": "C++", "48": "Agile", "49": "Scrum", "50": "MVC", "51": "Singleton", "52": "Factory",
    "53": "Unit Testing", "54": "Integration Testing", "55": "Selenium", "56": "AWS", "57": "Azure",
    "58": "Google Cloud", "59": "EC2", "60": "S3", "61": "Lambda", "62": "Terraform", "63": "CloudFormation",
    "64": "VPC", "65": "Load Balancers", "66": "DNS", "67": "IAM", "68": "Security Groups", "69": "Encryption",
    "70": "Vue.js", "71": "Django", "72": "GraphQL", "73": "Jenkins", "74": "Ansible", "75": "Puppet",
    "76": "Nagios", "77": "Prometheus", "78": "Grafana", "79": "Bitbucket", "80": "Unity", "81": "Unreal Engine",
    "82": "OpenGL", "83": "DirectX", "84": "Havok", "85": "Bullet", "86": "Multiplayer game networking",
    "87": "APIs", "88": "Windows", "89": "Linux", "90": "MacOS", "91": "TCP/IP", "92": "PC assembly",
    "93": "Peripheral troubleshooting", "94": "OS issues", "95": "Application errors", "96": "TeamViewer",
    "97": "Remote Desktop"
}
def prepare_data():
    # Create DataFrame
    data = {
        'current_skills': [
            [1, 2, 3, 4, 6, 7, 11, 14, 16],  # Full Stack Software Development
            [19, 20, 21, 22, 23, 24, 25, 27, 28],  # Data Science
            [32, 35, 36, 37, 38, 39, 40, 42, 43],  # Machine Learning Engineering
            [4, 8, 9, 10, 12, 15, 17, 18, 19],  # Web Development
            [2, 6, 7, 11, 13, 20, 22, 23, 29],  # Cloud Computing
            [16, 18, 20, 25, 28, 30, 31, 32, 33],  # Artificial Intelligence
            [3, 5, 9, 10, 12, 14, 17, 21, 23],  # Cybersecurity
            [8, 13, 15, 18, 24, 26, 27, 30, 35],  # DevOps Engineering
            [9, 11, 16, 19, 21, 25, 28, 34, 36],  # Software Engineering
            [10, 12, 14, 17, 22, 24, 26, 29, 31],  # Data Engineering
        ],
        'aptitude_score': [90, 85, 95, 80, 88, 92, 87, 83, 89, 91],  # Example aptitude scores
        'math_marks': [99, 92, 95, 85, 88, 91, 90, 87, 89, 94],      # Example math marks
        'science_marks': [100, 95, 98, 88, 90, 93, 89, 85, 92, 96],  # Example science marks
        'interests_goals': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],  # Example interests/goals
        'domain': [
            'Full Stack Software Development',
            'Data Science',
            'Machine Learning Engineering',
            'Web Development',
            'Cloud Computing',
            'Artificial Intelligence',
            'Cybersecurity',
            'DevOps Engineering',
            'Software Engineering',
            'Data Engineering',
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Convert domains to numeric labels
    le = LabelEncoder()
    df['domain_encoded'] = le.fit_transform(df['domain'])
    
    # Binarize the skills column
    mlb = MultiLabelBinarizer()
    skills_binarized = mlb.fit_transform(df['current_skills'])
    
    # Combine all features (skills and additional data)
    X = np.hstack([
        skills_binarized,
        df[['aptitude_score', 'math_marks', 'science_marks', 'interests_goals']].values
    ])
    
    y = df['domain_encoded']
    
    return X, y, le, mlb


# Build and train neural network model
def build_nn_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Main function to train the model
def train_model():
    X, y, label_encoder, mlb = prepare_data()
    
    num_classes = len(label_encoder.classes_)
    
    model = build_nn_model(X.shape[1], num_classes)
    model.fit(X, y, epochs=20, batch_size=2, verbose=1, validation_split=0.2)
    
    model.save('model.h5')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(mlb, 'mlb.pkl')
    print("Model and label encoder saved.")


# Load and predict using the model
def predict_recommendation(user_skills, model, label_encoder, mlb):
    user_skills_binarized = mlb.transform([user_skills])
    
    nn_predictions = model.predict(user_skills_binarized)
    nn_prediction_class = np.argmax(nn_predictions, axis=1)
    
    recommended_domain = label_encoder.inverse_transform([nn_prediction_class[0]])[0]
    return recommended_domain

# Training the model (Run once to train and save the model)
train_model()

def load_models():
    model = tf.keras.models.load_model('model.h5')
    label_encoder = joblib.load('label_encoder.pkl')
    mlb = joblib.load('mlb.pkl')
    return model, label_encoder, mlb
def preprocess_user_input(currentSkills, aptitudeScore, mathMarks, scienceMarks, interestsGoals):
    # Convert string IDs to integers for skills
    currentSkills = [int(skill_id) for skill_id in currentSkills]
    
    # Convert other inputs to numeric values
    aptitudeScore = int(aptitudeScore)
    mathMarks = int(mathMarks)
    scienceMarks = int(scienceMarks)
    interestsGoals = int(interestsGoals)
    
    return currentSkills, aptitudeScore, mathMarks, scienceMarks, interestsGoals
def predict_recommendation(user_skills: list[int], aptitude_score: int, math_marks: int, science_marks: int, interests_goals: int, model, label_encoder, mlb):
    # Preprocess user input
    skills_binarized = mlb.transform([user_skills])
    user_input = np.hstack([
        skills_binarized,
        np.array([[aptitude_score, math_marks, science_marks, interests_goals]])
    ])

    # Ensure the input shape matches the model's expected input shape
    if user_input.shape[1] != model.input_shape[1]:
        raise ValueError(f"Input shape mismatch: Expected {model.input_shape[1]}, but got {user_input.shape[1]}")

    # Make a prediction
    nn_predictions = model.predict(user_input)
    predicted_domain_idx = np.argmax(nn_predictions, axis=1)
    predicted_domain = label_encoder.inverse_transform(predicted_domain_idx)

    return predicted_domain[0]
