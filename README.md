# 📰 Fake News Detection using NLP & Machine Learning

## 📌 Project Overview
This project focuses on detecting **fake news articles** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
The model classifies news as **Real** or **Fake** based on textual content.

---

## 🎯 Objective
- To analyze news text data
- To clean and preprocess text using NLP
- To build a machine learning model that predicts fake news accurately

---

## 🗂️ Dataset
Dataset used from Kaggle (ISOT Fake News Dataset):

- `Fake.csv` – Fake news articles
- `True.csv` – Real news articles

Each record contains:
- Title
- Text
- Subject
- Date
- Label (0 = Fake, 1 = Real)

---

## 🛠️ Tools & Technologies Used
- Python 3
- Pandas
- NumPy
- Regular Expressions (re)
- Scikit-learn
- NLP (Text Cleaning & Vectorization)
- Git & GitHub
- VS Code

---

## 🔄 Project Workflow
1. Load and merge datasets
2. Data cleaning (lowercasing, removing punctuation, URLs, numbers)
3. Exploratory Data Analysis (EDA)
4. Feature extraction using TF-IDF Vectorizer
5. Train-test split
6. Model training (Logistic Regression)
7. Model evaluation
8. Sample prediction

---

## 📊 Sample Output
```text
Sample Prediction: FAKE NEWS
