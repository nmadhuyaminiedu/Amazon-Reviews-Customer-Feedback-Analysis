# 📦 Amazon Reviews – Customer Feedback Analysis

An end-to-end NLP project that processes Amazon customer reviews, detects sentiment (Positive, Neutral, Negative), and presents insights through a Power BI dashboard. Built to demonstrate full-stack data pipeline skills using Python, MongoDB, ML, and visualization.

---

## 🚀 Project Overview

This project automates the process of:
- Cleaning raw Amazon review data
- Classifying sentiment using a trained ML model
- Extracting insights via TF-IDF vectorization
- Visualizing results in Power BI

🔍 The final dashboard helps explore:
- Sentiment distribution
- Common keywords by sentiment
- Review count by rating
- Sample feedback tables

---

## 🛠️ Tech Stack & Tools
____________________________________________________________
| Area       | Tools & Libraries                           |
|------------|---------------------------------------------|
| Language   | Python 3.x                                  |
| Data Source| MongoDB (NoSQL)                             |
| Processing | pandas, nltk, spacy, scikit-learn, pymongo  |
| Modeling   | TF-IDF + Logistic Regression                |
| Dashboard  | Power BI                                    |
| Versioning | Git + GitHub                                |
|__________________________________________________________|
---

## 🗂️ Project Structure

```bash
scripts/
├── clean_data.py           # Cleans and labels text
├── vectorize_text.py       # TF-IDF transformation
├── train_model.py          # Trains sentiment classifier
├── load_data.py            # Loads data from MongoDB
├── predict_sentiment.py    # Predicts sentiment from input
├── export_to_csv.py        # Exports data to CSV for Power BI

data/
├── Reviews.csv             # Review Dataset
└── clean_reviews.csv       # Cleaned review dataset
