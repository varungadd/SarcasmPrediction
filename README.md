# SarcasmPrediction

This project focuses on detecting sarcasm in headlines using machine learning models. It includes various deep learning approaches like CNN, RNN, LSTM, and GRU, alongside techniques for text preprocessing and embedding.

Features
Preprocessing of text data (stopword removal, tokenization, and cleaning).
Implementation of word embeddings (GloVe and custom embeddings).
Multiple models: CNN, Bidirectional GRU, Bidirectional LSTM, and RNN.
Training, testing, and evaluation of sarcasm detection models.
WordCloud visualizations for sarcastic headlines.
Model saving for future predictions.
Dataset
The dataset used in this project is the Sarcasm Headlines Dataset, which consists of news headlines labeled as sarcastic (1) or not sarcastic (0).

Download the dataset from Sarcasm Headlines Dataset.

Prerequisites
Install the following Python packages before running the project:

numpy
pandas
tensorflow
matplotlib
wordcloud
scikit-learn
imbalanced-learn
nltk
To install all dependencies, run:

bash
Copy
Edit
pip install -r requirements.txt
Project Workflow
1. Data Preprocessing
Headlines are cleaned, tokenized, and stopwords are removed.
Headlines are then converted to sequences using Tokenizer and padded using pad_sequences.
2. Word Embedding
Pre-trained GloVe embeddings (glove.6B.100d.txt) are used to initialize word vectors.
Custom embeddings are generated if GloVe is not available.
3. Models
CNN: Convolutional Neural Networks for text feature extraction.
GRU: Bidirectional Gated Recurrent Units for sequential data modeling.
LSTM: Bidirectional Long Short-Term Memory networks.
RNN: A vanilla Recurrent Neural Network implementation.
4. Evaluation
Models are evaluated using accuracy, precision, recall, and loss plots.
WordCloud visualizations provide insights into the most frequent sarcastic words.
Usage
Train a Model
Run the corresponding code cells in the script to train the desired model.

Make Predictions
Use the predict_sarcasm() function to predict sarcasm on new headlines:

python
Copy
Edit
result = predict_sarcasm("Your sarcastic sentence here.")
print(result)
Example
python
Copy
Edit
predict_sarcasm("I love when my internet doesn't work!")
Results
The project achieves high accuracy with CNN and GRU models. Validation accuracy ranges between 80% to 85%, depending on the model and parameters used.

Directory Structure
css
Copy
Edit
Sarcasm_Detection/
├── Sarcasm_Headlines_Dataset.json
├── Models/
│   ├── cnn_model.h5
│   ├── gru_model.h5
│   ├── lstm_model.h5
│   └── tokenizer.pickle
├── Embeddings/
│   └── glove.6B.100d.txt
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── predictions.py
└── README.md
Visualizations
Training and Validation Accuracy


WordCloud for Sarcastic Headlines
