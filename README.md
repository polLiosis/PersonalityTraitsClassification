# Multi-Label Classification of Big Five Personality Traits

## 1. Introduction
This repository contains the code and resources for an MSc project focusing on predicting the Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) from text data. The project leverages a **multi-label classification** approach, where each trait is modeled as a separate binary label.

## 2. Dataset & Embeddings
- **Dataset**: [essays.zip](http://web.archive.org/web/20160519045708/http:/mypersonality.org/wiki/doku.php?id=wcpr13)
- **Embeddings**: 100 dim GloVe [Embeddings](https://nlp.stanford.edu/projects/glove/)

## 3. Project Overview
People often express their personality through their writing style, choice of words, and thematic content. By analyzing textual data, we can approximate which traits are more likely present in an individual. This project:
- Uses **Natural Language processing (NLP)** techniques to preprocess textual essays.
- Trains **multi-label classification** models to simultaneously predict five binary traits.
- Evaluates performance of models using metrics tailored to multi-label scenarios (e.g., **Macro/micro-F1**, **Multilabel Accuracy** etc.).

The Jupyter Notebook includes the following main sections:

- **Library Imports & GPU Configuration:**  
  Sets up logging, configures GPU memory usage, and imports necessary libraries (e.g., `numpy`, `pandas`, `tensorflow`, `gensim`, `matplotlib`, `seaborn`, etc.).

- **Dataset Preprocessing:**  
  Loads the dataset from `Datasets/essays.csv`, cleans the text (e.g., lowercasing, removing punctuation and digits), and renames label columns to reflect the Big Five personality traits.

- **Tokenization and Padding:**  
  Uses Keras' `Tokenizer` to tokenize the preprocessed text, converts texts to sequences, and pads them to a maximum sequence length determined by a percentile of sequence lengths.

- **Embedding Techniques:**  
  - **Word2Vec:** Trains a Word2Vec model on tokenized sentences and builds an embedding matrix.  
  - **GloVe:** Loads pre-trained GloVe embeddings from `Datasets/glove.6B.100d.txt` and constructs an embedding matrix.

- **Model Building and Training:**  
  Builds and trains multiple deep learning models for multi-label classification:
  - **LSTM Models:** Two variants are implemented—one using Word2Vec embeddings and the other using GloVe embeddings.  
  - **Transformer Model:** Utilizes multi-head attention and other layers to capture sequence-level information for classification.
  
  Each model is compiled with binary crossentropy loss, optimized using Adam, and trained using callbacks such as EarlyStopping and ReduceLROnPlateau.

- **Evaluation:**  
  Evaluates model performance on a test set using metrics such as accuracy, precision, recall, and F1-scores. The notebook also generates visualizations (loss/accuracy plots and confusion matrices) to analyze the predictions of each model.
