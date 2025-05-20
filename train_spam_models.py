import pandas as pd
import re
import string
import requests
import io
import zipfile
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

print("Starting spam detection model training...")

# Dictionary to store our models
models = {
    'SMS': LogisticRegression(max_iter=1000),
    'Email': LogisticRegression(max_iter=1000)
}

# SMS Spam Detection
print("Training SMS spam model...")

# Load SMS data
try:
    # Method 1: Try to load from local file first
    df_sms = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
except FileNotFoundError:
    # Method 2: If local file not found, download from URL
    print("Local SMS file not found, downloading from URL...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    response = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    with z.open('SMSSpamCollection') as f:
        df_sms = pd.read_csv(f, sep='\t', header=None, names=['label', 'message'], encoding='latin-1')
    df_sms.to_csv('SMSSpamCollection', sep='\t', index=False, header=False)

# Preprocess SMS data
df_sms['label'] = df_sms['label'].map({'ham': 0, 'spam': 1})

def preprocess_txt(txt):
    txt = str(txt).lower()
    txt = re.sub(r'\d+', '', txt)
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = txt.strip()
    return txt

df_sms['message'] = df_sms['message'].apply(preprocess_txt)

# TF-IDF for SMS
tfidf_sms = TfidfVectorizer(max_features=5000)
X_sms = tfidf_sms.fit_transform(df_sms['message']).toarray()
y_sms = df_sms['label']

# Split SMS data
X_sms_train, X_sms_test, y_sms_train, y_sms_test = train_test_split(
    X_sms, y_sms, test_size=0.2, random_state=42)

# Train SMS model
models['SMS'].fit(X_sms_train, y_sms_train)
sms_accuracy = models['SMS'].score(X_sms_test, y_sms_test)
print(f"SMS spam model accuracy: {sms_accuracy:.4f}")

# Save SMS model and vectorizer
joblib.dump(models['SMS'], 'spam_model_sms.pkl')
joblib.dump(tfidf_sms, 'tfidf_vectorizer_sms.pkl')

# Email Spam Detection
print("Training Email spam model...")

def load_email_data():
    try:
        # Method 1: Try to load from local file first
        return pd.read_csv('spambase.data', header=None)
    except FileNotFoundError:
        # Method 2: If local file not found, download from URL
        print("Local email file not found, downloading from URL...")
        email_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        df = pd.read_csv(email_url, header=None)
        df.to_csv('spambase.data', index=False, header=False)
        return df

# Column names for the Spambase dataset
email_columns = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
    'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order',
    'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people',
    'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business',
    'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your',
    'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl',
    'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs',
    'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415',
    'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
    'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
    'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu',
    'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(',
    'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average',
    'capital_run_length_longest', 'capital_run_length_total', 'is_spam'
]

# Load and process Email data
email_df = load_email_data()
email_df.columns = email_columns

# Scale features for Email model
scaler = StandardScaler()
X_email = scaler.fit_transform(email_df.drop('is_spam', axis=1))
y_email = email_df['is_spam']

# Split Email data
X_email_train, X_email_test, y_email_train, y_email_test = train_test_split(
    X_email, y_email, test_size=0.2, random_state=42)

# Train Email model
models['Email'].fit(X_email_train, y_email_train)
email_accuracy = models['Email'].score(X_email_test, y_email_test)
print(f"Email spam model accuracy: {email_accuracy:.4f}")

# Save Email model and scaler
joblib.dump(models['Email'], 'spam_model_email.pkl')
joblib.dump(scaler, 'email_scaler.pkl')

print("Training completed successfully!")
print(f"SMS Model saved as 'spam_model_sms.pkl' (Accuracy: {sms_accuracy:.4f})")
print(f"Email Model saved as 'spam_model_email.pkl' (Accuracy: {email_accuracy:.4f})")