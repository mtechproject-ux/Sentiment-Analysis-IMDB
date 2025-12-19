# Sentiment Analysis of Product Reviews (IMDB)

## 📌 Overview
This project performs sentiment analysis on text reviews using Natural Language Processing (NLP) techniques. Reviews are classified into Positive and Negative sentiments.

## 🛠 Technologies Used
- Python
- NLTK
- TF-IDF
- Naïve Bayes

## 📂 Dataset
- IMDB Dataset
- 5,000 cleaned reviews sampled from the original dataset

## ⚙ Methodology
- Text preprocessing: tokenization, stop-word removal, lemmatization
- Feature extraction using TF-IDF (unigrams + bigrams)
- Classification using Multinomial Naïve Bayes

## 📊 Results
- F1-score: **0.86**
- Balanced performance for both positive and negative classes

## ▶ How to Run
```bash
pip install -r requirements.txt
cd src
python train_model.py
