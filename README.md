#  Spam Classifier

A simple **machine learning web app** that classifies SMS messages as **Spam or Ham** using **Logistic Regression & TF-IDF**. Built with **Python, Flask, and Scikit-Learn**. 
---
## Features  
- Text preprocessing & TF-IDF vectorization  
- Logistic Regression for classification  
- Flask web app for user interaction 

---
##  Project Structure  
sms-spam-classifier  
│── static/           
│── templates/          
│── SMSSpamCollection 
│── spam_model.pkl       
│── tfidf_vectorizer.pkl
│── train_model.py       
│── app.py               


---
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/OmarDawoud4/Logistic.git
cd Logistic
```
### **Train The Model**  
```bash
python train_model.py
```
### **Run the Flask App**
```bash
python app.py

```
The app will start running at local host
