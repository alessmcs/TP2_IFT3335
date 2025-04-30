# Viviane BINET (), Alessandra MANCAS (20249098)

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib
import transformers
import re  # regex library

from transformers import model_addition_debugger


### PARTIE 1 ###

# Prétraitement
def pretraitement(df):
    apply_regex = lambda s: (re.sub('[^A-Za-z0-9 ]+', '', s)).lower()
    return df.apply(apply_regex)

# Bag of words
# https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c/
def bag_of_words(content):
    vectorizer = CountVectorizer(max_features=5000)
    count_matrix = vectorizer.fit_transform(content)
    count_array = count_matrix.toarray()
    df = pd.DataFrame(data=count_array, columns=vectorizer.get_feature_names_out())

    return df

# TF-IDF
def tf_idf(contents):
    vectorizer = TfidfVectorizer(max_features=5000)
    count_matrix = vectorizer.fit_transform(contents)
    count_array = count_matrix.toarray()
    df = pd.DataFrame(data=count_array, columns=vectorizer.get_feature_names_out())
    return df

# Entraîner modèles (Logistic regression, Random Forest, MLP)
# https://thecleverprogrammer.com/2024/02/19/compare-multiple-machine-learning-models/
def train_model(model, inputs, labels, test_inputs, test_labels):
    # x : input, y: label
    classifier = model
    classifier.fit(inputs, labels)

    y_pred = classifier.predict(test_inputs)
    print(y_pred)

    # tout calculer p/r à l'ensemble de test
    scores = cross_val_score(classifier, test_inputs, test_labels, cv=5)
    accuracy = accuracy_score(test_labels, y_pred)
    f1_score = metrics.f1_score(test_labels, y_pred)

    return scores, accuracy, f1_score

def to_embedding(text):

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text)
    return embeddings



def main():

    df = pd.read_csv('TP2_code/spam_train.csv')

    # Preprocess the data
    df['text'] = pretraitement(df['text'])
    df['label'] = pd.to_numeric(df['label'], downcast='integer', errors='coerce')
    print(df.head())


    fm = bag_of_words(df['text'])
    fm_train, fm_test, y1_train, y1_test = train_test_split(fm, df['label'], random_state=42)
    # print(fm.head)

    tfidf = tf_idf(df['text'])
    tfidf_train, tfidf_test, y2_train, y2_test = train_test_split(tfidf, df['label'], random_state=42)
    # print(tfidf_train.head)

    # todo: choose parameters
    models = {
        "LR": linear_model.LogisticRegression(),
        "RF": RandomForestClassifier(),
        "MLP": MLPClassifier()
    }

    vector = []

    # for i in tqdm(range(len(df["text"]))):
    for i in tqdm(range(20)):
        vector.append(to_embedding(df["text"][i]))

    print(vector[0])

    for model in models:
        print("training model " + model)

        classifier = models[model]
        bow_scores, bow_accuracy, bow_f1 = train_model(classifier, fm_train, y1_train, fm_test, y1_test)
        tfidf_scores, tfidf_accuracy, tfidf_f1 = train_model(classifier, tfidf_train, y2_train, tfidf_test, y2_test)

        print('bow scores', bow_scores, 'avg')
        print('bow accuracy', bow_accuracy)
        print('bow f1', bow_f1, '\n')

        print('tfidf scores', tfidf_scores)
        print('tfidf accuracy', tfidf_accuracy)
        print('tfidf f1', tfidf_f1, '\n')




# Entry point
if __name__ == "__main__":
    main()