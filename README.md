Quora Duplicate Question Detection
Project Overview
This project aims to build a machine learning model to identify duplicate questions on Quora. By detecting duplicate questions, Quora can consolidate answers, reduce redundancy, and improve the user experience. The model predicts whether a pair of questions is duplicate or not, enabling Quora to provide users with existing answers for commonly asked questions.

Problem Statement
Given a pair of questions, the task is to predict if they are semantically similar or duplicates. This will help Quora in organizing its content better and making it easier for users to find the best answers.

Dataset
The dataset used in this project is from Kaggle's Quora Question Pairs. It contains pairs of questions and a target label indicating if they are duplicates.

Dataset Features:
id: unique ID for each question pair
qid1, qid2: unique IDs for the first and second questions
question1, question2: the actual questions in text format
is_duplicate: label (1 if questions are duplicates, 0 otherwise)
Approach
Data Preprocessing:

Removed missing values.
Cleaned text by removing special characters, stopwords, and applying stemming.
Feature Engineering:

Created features based on text similarities, such as:
Word count differences
Character count differences
Common words in both questions
Utilized TF-IDF vectorization, cosine similarity, and word embeddings.
Modeling:

Experimented with models including:
Logistic Regression
Support Vector Machine (SVM)
Random Forest
Siamese Neural Networks with LSTMs for deep learning.
Evaluated models based on F1 Score, Precision, Recall, and AUC-ROC.
Model Evaluation & Optimization:

Selected the best-performing model and tuned hyperparameters to optimize performance.
Cross-validated the model to ensure its generalizability.
Results
Achieved high accuracy and strong F1 score in predicting duplicate questions, making the model effective for real-world usage on Quora's platform.

Requirements
Python 3.x
Jupyter Notebook
Libraries: pandas, numpy, scikit-learn, nltk, tensorflow (for deep learning models), re, matplotlib, seaborn
