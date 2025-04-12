# Toxic Comment Classification Project

This project focuses on detecting various forms of toxic comments using deep learning and NLP techniques. The dataset used is from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). The goal is to identify different types of toxicity such as **toxic**, **severe toxic**, **obscene**, **threat**, **insult**, and **identity hate** in user comments.

---

## Project Structure

The project consists of two main Python files:

- **File 1**: Implements a basic LSTM-based model with Word2Vec embeddings and tracks batch-wise accuracy.
- **File 2**: Builds a more complex BiLSTM architecture with regularization and dropout, and compares it to a baseline SGD classifier using One-vs-Rest strategy.

---

## Key Concepts

- **Word Embeddings**: Used Word2Vec to convert textual data into 300-dimensional dense vectors.
- **Deep Learning Models**:
  - File 1: Basic Bidirectional LSTM with max pooling.
  - File 2: Enhanced LSTM stack with dropout, batch normalization, and regularization.
- **Baseline Model**: Scikit-learn's `SGDClassifier` using `OneVsRestClassifier` to handle multi-label classification.
- **Evaluation Metrics**:
  - Accuracy (Training, Validation)
  - ROC AUC Score
- **Visualization**:
  - Accuracy tracking every 100 training examples (File 1).
  - Real-time accuracy plotting for mini-batch SGD (File 2).
  - Epoch-wise accuracy and loss curves.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Gensim (for Word2Vec)
- Pandas & NumPy
- Matplotlib

---

## Dataset

- **Source**: [Kaggle Jigsaw Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- **Features**: `comment_text`
- **Labels**:  
  - `toxic`  
  - `severe_toxic`  
  - `obscene`  
  - `threat`  
  - `insult`  
  - `identity_hate`

---

## Model Performance

### Deep Learning Model (File 1)

- Trained using TPU (if available)  
- Batch-wise accuracy tracking using a custom Keras callback  
- Final ROC AUC Score evaluated on a test split

### Deep Learning + Regularization (File 2)

- Used Spatial Dropout, L2 Regularization, BatchNorm  
- Showcased training/validation accuracy and loss  
- ROC AUC Score used as evaluation metric

### Baseline SGD Classifier

- Trained using mini-batch online learning  
- Real-time accuracy visualization after every batch

---

## How to Run

Note: This project was built and tested on **Kaggle Notebooks**.

1. Download the dataset from the [Kaggle challenge page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).
2. Place the ZIP file in the working directory.
3. Run the notebook or Python files in a Kaggle or Colab environment.
4. Visualizations will be shown inline.

---

## Results

The models were able to effectively classify multi-label toxic categories. Word2Vec embeddings helped boost performance, and LSTM-based architectures showed strong learning capabilities. Additional regularization (in File 2) further stabilized training.

---

## Future Improvements

- Try pretrained embeddings like GloVe or FastText.
- Use transformer-based models (e.g., BERT or RoBERTa).
- Improve handling of class imbalance (e.g., focal loss, oversampling).
- Implement attention mechanisms.

---

## Author

Lubaina Malvi, Zoe Langelaan, Sonia Popovic

---
