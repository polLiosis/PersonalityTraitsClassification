# Multi-Label Classification of Big Five Personality Traits

## 1. Introduction
This repository contains the code and resources for an MSc project focusing on predicting the Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) from text data. The project leverages a **multi-label classification** approach, where each trait is modeled as a separate binary label.

## 2. Problem Statement
People often express their personality through their writing style, choice of words, and thematic content. By analyzing textual data, we can approximate which traits are more likely present in an individual. This project:
- Uses **Natural Language processing (NLP)** techniques to preprocess textual essays.
- Trains a **multi-label classification** model to simultaneously predict five binary traits.
- Evaluates performance using metrics tailored to multi-label scenarios (e.g., **Macro/micro-F1**, **Multilabel Accuracy** etc.).

## 3. Dataset & Embeddings
- **Dataset**: [essays.zip](http://web.archive.org/web/20160519045708/http:/mypersonality.org/wiki/doku.php?id=wcpr13)
- **Embeddings**: 100 dim GloVe [Embeddings](https://nlp.stanford.edu/projects/glove/)
