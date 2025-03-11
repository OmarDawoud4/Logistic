from flask import Flask, request, render_template
import joblib
import re
import string

app = Flask(__name__)
model = joblib.load('spam_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

def preprocess_txt(txt):
    txt = txt.lower()
    txt = re.sub(r'\d+', '', txt)
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    txt = txt.strip()
    return txt

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        message = request.form['message']
        if message:

            processed_message = preprocess_txt(message)
            vectorized_message = tfidf.transform([processed_message]).toarray()
            pred = model.predict(vectorized_message)[0]
            prediction = 'Spam' if pred == 1 else 'Ham'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)