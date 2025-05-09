# ChatGPT Sentiment Analysis (LSTM + GloVe Embedding)

This project analyzes public sentiment toward ChatGPT using tweets. It implements a deep learning approach using **LSTM networks** enhanced by **pre-trained GloVe word embeddings**.

---

## Project Overview

Classify ChatGPT-related tweets into **three sentiment categories**:

* `good` (positive sentiment)
* `neutral`
* `bad` (negative sentiment)

---

## Dataset

* **Size**: \~219,000 tweets
* **Columns**:

  * `tweets`: raw tweet text
  * `labels`: sentiment label (`good`, `neutral`, `bad`)
* **Source**: Pre-collected and cleaned tweet dataset

---

## Methodology

### LSTM with GloVe Embeddings

* **Text Preprocessing**:

  * Lowercasing, punctuation removal, stopword removal
  * Tokenization using Keras
* **Embedding**:

  * Pre-trained GloVe vectors (100-dimensional)
  * Embedding layer was initially frozen (non-trainable)
* **Model Architecture**:

  * Embedding Layer (GloVe)
  * LSTM Layer with 128 units
  * Dense layers with ReLU activation and Dropout for regularization
  * Output layer with Softmax for 3-class classification
* **Training Details**:

  * 10 epochs, batch size of 128
  * Validation split of 10%
  * Class weights applied to address label imbalance

---

## Results

* **Training Accuracy**: Peaked around \~39% with class weights applied
* **Validation Accuracy**: Plateaued at \~49.5%
* **Test Set Performance**:

  * The model **predicted only the 'neutral' class**, resulting in 0 precision/recall for `bad` and `good`
  * Likely due to:

    * Freezing the embedding layer
    * Insufficient model capacity or tuning
    * Severe class imbalance

### Classification Report (Truncated)

```
              precision    recall  f1-score   support

         bad       0.00      0.00      0.00    21559
     neutral       0.49      1.00      0.66    11198
        good       0.00      0.00      0.00    11202

    accuracy                           0.49    43859
   macro avg       0.16      0.33      0.22    43859
weighted avg       0.24      0.49      0.32    43859
```

---

## Tools & Libraries

* Python
* Pandas, NLTK, Scikit-learn
* TensorFlow / Keras
* Matplotlib, Seaborn

---

## Future Improvements

* **Unfreeze embedding layer** to allow fine-tuning of word vectors
* Increase LSTM capacity or try BiLSTM
* Add more epochs and learning rate scheduling
* Apply advanced methods: GRU, Transformer-based models (e.g., BERT)
* Conduct hyperparameter tuning

---

## Getting Started

### Install Requirements

```bash
pip install -r requirements.txt
```

### Run Notebook (Google Colab Recommended)

* `LSTM_GloVe_Model.ipynb`

---

## Author

This project was developed as part of a portfolio on deep learning-based NLP using real-world social media data.

---

## License

MIT License
