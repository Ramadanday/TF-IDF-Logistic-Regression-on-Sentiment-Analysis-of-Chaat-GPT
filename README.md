# ChatGPT Sentiment Analysis (TF-IDF + Logistic Regression)

This project analyzes public sentiment toward ChatGPT using a traditional NLP pipeline. It uses **TF-IDF vectorization** combined with a **Logistic Regression** classifier to categorize tweets into sentiment classes.

---

## Project Overview

Classify tweets mentioning ChatGPT into three sentiment categories:

* `good` (positive sentiment)
* `neutral`
* `bad` (negative sentiment)

---

## Dataset

* **Size**: \~219,000 tweets
* **Columns**:

  * `tweets`: raw tweet text
  * `labels`: sentiment label (`good`, `neutral`, `bad`)
* **Source**: Pre-collected tweet dataset on ChatGPT

---

## Methodology

### TF-IDF + Logistic Regression

* **Text Preprocessing**:

  * Lowercasing, punctuation and stopword removal
  * Tokenization using NLTK
* **Feature Extraction**:

  * TF-IDF Vectorizer with `max_features=5000`
  * Converts tweets into numerical vectors based on word importance
* **Model Training**:

  * Logistic Regression from Scikit-learn
  * 80/20 train-test split with stratified labels

---

## Results

* **Accuracy**: **83%** on the test set
* **F1-Score per Class**:

  * `bad`: 0.91
  * `neutral`: 0.69
  * `good`: 0.81
* **Observations**:

  * Model performs well on polarized sentiment
  * Neutral sentiment is more difficult to capture

---

## Tools & Libraries

* Python
* Pandas, NLTK, Scikit-learn
* Matplotlib, Seaborn

---

## Future Improvements

* Use n-gram features to capture short phrases
* Try alternative classifiers (e.g., SVM, Random Forest)
* Handle class imbalance with resampling or class weights
* Visualize common words and sentiment trends

---

## Getting Started

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Notebook (Google Colab Recommended)

* `TFIDF_LogisticRegression.ipynb`

---

## Author

This project was developed as part of a portfolio on NLP and sentiment analysis using real-world social media data.

---

## License

MIT License
