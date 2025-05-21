from flask import Flask, request, jsonify, render_template
import joblib
import re
import string
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models and preprocessing tools
try:
    # SMS spam detection
    sms_model = joblib.load('spam_model_sms.pkl')
    tfidf_sms = joblib.load('tfidf_vectorizer_sms.pkl')

    # Email spam detection
    email_model = joblib.load('spam_model_email.pkl')
    email_scaler = joblib.load('email_scaler.pkl')

    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run train_spam_models.py first to generate the models.")


def preprocess_sms(txt):
    """Preprocess SMS text - convert to lowercase, remove numbers, punctuation and strip whitespace"""
    txt = str(txt).lower()
    txt = re.sub(r'\d+', '', txt)
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = txt.strip()
    return txt


def extract_email_features(email_text):
    """Extract features from email text similar to the spambase dataset structure"""
    # This is a simplified version - in a production system you would need a more sophisticated feature extraction
    email_text = email_text.lower()

    # Initialize features dictionary with zeros
    features = {}

    # Word frequencies (57 features in original dataset)
    word_list = [
        'make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order',
        'mail', 'receive', 'will', 'people', 'report', 'addresses', 'free', 'business',
        'email', 'you', 'credit', 'your', 'font', '000', 'money', 'hp', 'hpl',
        'george', '650', 'lab', 'labs', 'telnet', '857', 'data', '415',
        '85', 'technology', '1999', 'parts', 'pm', 'direct', 'cs', 'meeting',
        'original', 'project', 're', 'edu', 'table', 'conference'
    ]

    for word in word_list:
        # Count occurrences and normalize by total words
        count = email_text.count(word)
        total_words = len(email_text.split())
        features[f'word_freq_{word}'] = count / max(1, total_words) * 100  # as percentage

    # Character frequencies (6 features)
    char_list = [';', '(', '[', '!', '$', '#']
    for char in char_list:
        count = email_text.count(char)
        total_chars = len(email_text)
        features[f'char_freq_{char}'] = count / max(1, total_chars) * 100

    # Capital run length statistics (3 features)
    capital_runs = re.findall(r'[A-Z]+', email_text)

    if capital_runs:
        features['capital_run_length_average'] = sum(len(run) for run in capital_runs) / len(capital_runs)
        features['capital_run_length_longest'] = max(len(run) for run in capital_runs) if capital_runs else 0
        features['capital_run_length_total'] = sum(len(run) for run in capital_runs)
    else:
        features['capital_run_length_average'] = 0
        features['capital_run_length_longest'] = 0
        features['capital_run_length_total'] = 0

    # Create feature vector in the same order as the training data
    feature_names = [
        *[f'word_freq_{word}' for word in word_list],
        *[f'char_freq_{char}' for char in char_list],
        'capital_run_length_average',
        'capital_run_length_longest',
        'capital_run_length_total'
    ]

    feature_vector = [features[name] for name in feature_names]
    return np.array(feature_vector).reshape(1, -1)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_sms', methods=['POST'])
def predict_sms():
    try:
        data = request.get_json()
        message = data['message']

        # Preprocess the message
        processed_msg = preprocess_sms(message)

        # Vectorize
        X = tfidf_sms.transform([processed_msg]).toarray()

        # Predict
        prediction = sms_model.predict(X)[0]
        probability = sms_model.predict_proba(X)[0][1]

        return jsonify({
            'message': message,
            'is_spam': bool(prediction),
            'spam_probability': float(probability),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })


@app.route('/predict_email', methods=['POST'])
def predict_email():
    try:
        data = request.get_json()
        email_text = data['email']

        # Extract features from email
        features = extract_email_features(email_text)

        # Scale features
        scaled_features = email_scaler.transform(features)

        # Predict
        prediction = email_model.predict(scaled_features)[0]
        probability = email_model.predict_proba(scaled_features)[0][1]

        return jsonify({
            'is_spam': bool(prediction),
            'spam_probability': float(probability),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
