import pandas as pd
import numpy as np
import re
import string
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

def preprocess_txt(txt):
    txt = txt.lower()
    txt = re.sub(r'\d+', '', txt)
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = txt.strip()
    return txt

df['message'] = df['message'].apply(preprocess_txt)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['message']).toarray()
y = df['label']

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, 'spam_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("training is done ")