This project focuses on detecting various forms of toxic comments using deep learning and NLP techniques. The dataset used is from the Jigsaw Toxic Comment Classification Challenge. The goal is to identify different types of toxicity such as toxic, severe toxic, obscene, threat, insult, and identity hate in user comments.

The project consists of two main Python files:
File 1: Implements a basic LSTM-based model with Word2Vec embeddings and tracks batch-wise accuracy.
File 2: Builds a more complex BiLSTM architecture with regularization and dropout, and compares it to a baseline SGD classifier using One-vs-Rest strategy.

Key concepts:
1. Word Embeddings: Used Word2Vec to convert textual data into 300-dimensional dense vectors.

2. Deep Learning Models:
File 1: Basic Bidirectional LSTM with max pooling.
File 2: Enhanced LSTM stack with dropout, batch normalization, and regularization.
Baseline Model: Scikit-learn's SGDClassifier using OneVsRestClassifier to handle multi-label classification.

3. Evaluation Metrics:
Accuracy (Training, Validation)
ROC AUC Score

4. Visualization:
Accuracy tracking every 100 training examples (File 1).
Real-time accuracy plotting for mini-batch SGD (File 2).
Epoch-wise accuracy and loss curves.

Technologies Used:
Python
TensorFlow / Keras
Scikit-learn
Gensim (for Word2Vec)
Pandas & NumPy
Matplotlib

Source: Kaggle Jigsaw Dataset
Features: comment_text
Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate

Model Performance: 
Deep Learning Model (File 1):
Trained using TPU (if available)
Batch-wise accuracy tracking
Final ROC AUC Score evaluated on a test split

Deep Learning + Regularization (File 2):
Used Spatial Dropout, L2 Regularization, BatchNorm
Showcased training/validation accuracy and loss
ROC AUC Score used as evaluation metric

Baseline SGD Classifier:
Trained using mini-batch online learning
Real-time accuracy visualization after every batch

The models were able to effectively classify multi-label toxic categories. Word2Vec embeddings helped boost performance, and LSTM-based architectures showed strong learning capabilities. Additional regularization (in File 2) further stabilized training.


Authors: Lubaina Malvi, Zoe Langelaan, Sonia Popovic
