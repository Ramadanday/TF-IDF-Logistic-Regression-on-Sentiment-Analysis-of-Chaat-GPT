ChatGPT Sentiment Analysis (TF-IDF + Logistic Regression)

This project focuses on analyzing public sentiment toward ChatGPT using tweets. It uses a traditional machine learning approach: TF-IDF vectorization combined with a Logistic Regression classifier to classify tweets into sentiment categories.

Project Overview

Analyze tweets related to ChatGPT and classify them into three sentiment categories:

good (positive sentiment)

neutral

bad (negative sentiment)

Dataset

Size: ~219,000 tweets

Columns:

tweets: raw tweet text

labels: sentiment label (good, neutral, bad)

Source: Pre-collected and cleaned tweet dataset

Methodology

TF-IDF + Logistic Regression

Text Preprocessing:

Convert to lowercase

Remove punctuation and stopwords

Tokenize text

Feature Extraction:

TF-IDF Vectorizer with max_features=5000

Transform text data into numerical vectors based on word importance

Model Training:

Logistic Regression classifier from scikit-learn

80/20 train-test split

Results

Accuracy: 83% on test set

F1-Score by Class:

bad: 0.91

neutral: 0.69

good: 0.81

Insights:

Strong performance in identifying positive and negative sentiments

Neutral class was harder to classify, likely due to linguistic ambiguity

Tools & Libraries

Python

Pandas, NLTK, Scikit-learn

Matplotlib, Seaborn

Future Improvements

Address class imbalance with resampling or class weights

Use n-grams or additional metadata as features

Explore deep learning methods (e.g., LSTM, BERT)

Deploy the model with Streamlit or Flask for interactive use

Getting Started

Install Requirements

pip install -r requirements.txt

Run Notebook (Google Colab Recommended)

TFIDF_LogisticRegression.ipynb

Author

This project was developed as part of a portfolio on NLP and sentiment analysis using real-world social media data.

License

MIT License

