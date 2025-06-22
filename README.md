# Hate Speech Detection using NLP and Machine Learning

This project presents a robust hate speech detection system leveraging modern Natural Language Processing (NLP) techniques and machine learning models. With the rise of social media, automating the identification of hate speech is essential to create safer and more inclusive digital platforms.

---

## 🔍 Project Overview

- **Goal**: Classify social media text into Hate Speech (1) or Non-Hate Speech (0)
- **Models Used**: DistilBERT (best), LSTM, Naive Bayes
- **Dataset Size**: ~700,000+ labeled social media comments
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUROC

---

## Motivation

Social media often includes hate speech masked by emoticons, slang, or informal language. Our goal is to build models capable of accurately detecting such harmful content to support online moderation and compliance with platform guidelines.

---

## Workflow

1. **Dataset Curation & Preprocessing**
   - Text cleaning, duplicate removal, language detection (English-only)
   - Tokenization, punctuation stripping, stopword removal

2. **Exploratory Data Analysis (EDA)**
   - Label distribution, sentiment analysis, text length CDFs
   - Hypothesis testing and Chi-Square tests

3. **Model Training & Evaluation**
   - **DistilBERT**: Achieved the highest performance (F1-Score = 0.9123)
   - **LSTM**: Sequential model capturing word dependencies (F1-Score = 0.86)
   - **Naive Bayes**: Efficient baseline classifier (F1-Score = 0.78)

4. **Visualization & Metrics**
   - Confusion matrix, sentiment distribution, ROC curve
   - Precision, recall, and F1-score plots

---

## Performance Summary

| Model        | Accuracy | Precision | Recall | F1-Score |
|--------------|----------|-----------|--------|----------|
| Naive Bayes  | 0.7796   | 0.79      | 0.78   | 0.78     |
| LSTM         | 0.8607   | 0.86      | 0.86   | 0.86     |
| **DistilBERT** | **0.9118** | **0.9126** | **0.912** | **0.9123** |

---

## Requirements

Install the required Python libraries:

```bash
pip install pandas numpy nltk seaborn matplotlib sklearn transformers vaderSentiment langdetect
