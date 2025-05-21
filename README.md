#  Spam Classifier

A simple **machine learning web app** that classifies SMS messages as **Spam or Ham** using **Logistic Regression & TF-IDF**. Built with **Python, Flask, and Scikit-Learn**. 

---
## 📓 Google Colab Notebook

This repository contains a Google Colab notebook that you can open and run directly in your browser.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nnFDMksB7WwIJzj9pcn6aWtsbokjNcjA?usp=sharing#scrollTo=9t0dbTdAP-xW)

---
## 🌐 Live Demo
🚀 Hosted at:
[spam-classifier](https://spam-classifier.up.railway.app/)
You can access and interact with the spam clssifier modules directly through the hosted web interface.

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
python train_spam_models.py
```
### **Run the Flask App**
```bash
python app.py

```
The app will start running at local host
