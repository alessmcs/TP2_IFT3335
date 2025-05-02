# Viviane BINET (20244728), Alessandra MANCAS (20249098)
import os.path
import sys

import pandas as pd
import numpy as np
from collections.abc import Sequence
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn import svm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import matplotlib
import transformers
import re  # regex library

import mlens
from mlens.ensemble import SuperLearner


import sys

from transformers import model_addition_debugger


### PARTIE 1 ###

# Prétraitement
def pretraitement(df):
    apply_regex = lambda s: (re.sub('[^A-Za-z0-9 ]+', '', s)).lower()
    return df.apply(apply_regex)

# Bag of words
# https://towardsdatascience.com/basics-of-countvectorizer-e26677900f9c/
def bag_of_words(content, no_features):
    vectorizer = CountVectorizer(max_features=no_features)
    count_matrix = vectorizer.fit_transform(content)
    count_array = count_matrix.toarray()
    df = pd.DataFrame(data=count_array, columns=vectorizer.get_feature_names_out())

    return df

# TF-IDF
def tf_idf(contents, no_features):
    vectorizer = TfidfVectorizer(max_features=no_features)
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

    # tout calculer p/r à l'ensemble de test
    cross_val = np.mean(cross_val_score(classifier, test_inputs, test_labels, cv=5))
    accuracy = accuracy_score(test_labels, y_pred)
    f1_score = metrics.f1_score(test_labels, y_pred)

    return cross_val, accuracy, f1_score

def to_embedding(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts.tolist(), batch_size=32)  # Convert Series to list
    return embeddings


def super_learner(x_train, y_train, x_test, y_test, scorer):
    ensemble = SuperLearner(scorer = accuracy_score, random_state=42, verbose=True, folds=5)
    ensemble.add([LogisticRegression(), RandomForestClassifier(), svm.SVC()])
    ensemble.add_meta(LogisticRegression())

    ensemble.fit(x_train, y_train)
    y_pred = ensemble.predict(x_test)

    # test de performance
    accuracy = accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)

    weights = []

    return accuracy, f1_score, weights

def main():

    scores_global = {}

    df = pd.read_csv('spam_train.csv')

    # Preprocess the data
    df['text'] = pretraitement(df['text'])
    df['label'] = pd.to_numeric(df['label'], downcast='integer', errors='coerce')
    print(df.head())

    fm = bag_of_words(df['text'], 5000)
    fm_train, fm_test, y1_train, y1_test = train_test_split(fm, df['label'], random_state=42)
    # print(fm.head)

    tfidf = tf_idf(df['text'], 5000)
    tfidf_train, tfidf_test, y2_train, y2_test = train_test_split(tfidf, df['label'], random_state=42)
    # print(tfidf_train.head)

    # todo: choose parameters
    models = {
        "Logistic Regression": linear_model.LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "MultiLayer Perceptron": MLPClassifier()
    }

    vector = []
    max_size = 40
    df_small = df[0:max_size]

    for i in tqdm(range(max_size)):
        vector.append(to_embedding(df["text"][i]))

    print(vector[0])

    embed_train, embed_test, y3_train, y3_test = train_test_split(vector, df_small["label"], random_state=42)


    for model in models:
        print("training model " + model)

        # BoW & TF-IDF (partie 1)
        classifier = models[model]
        bow_cross_val, bow_accuracy, bow_f1 = train_model(classifier, fm_train, y1_train, fm_test, y1_test)
        tfidf_cross_val, tfidf_accuracy, tfidf_f1 = train_model(classifier, tfidf_train, y2_train, tfidf_test, y2_test)

        print('bow cross_val', bow_cross_val)
        print('bow accuracy', bow_accuracy)
        print('bow f1', bow_f1, '\n')

        print('tfidf cross_val', tfidf_cross_val)
        print('tfidf accuracy', tfidf_accuracy)
        print('tfidf f1', tfidf_f1, '\n')

        # Embeddings (partie 2)

        embed_cross_val, embed_accuracy, embed_f1 = train_model(classifier, embed_train, y3_train, embed_test, y3_test)
        print('embed_cross_val', embed_cross_val, "\n")

        scores_global[model] = [bow_cross_val, bow_accuracy, bow_f1, tfidf_cross_val, tfidf_accuracy, tfidf_f1, embed_cross_val, embed_accuracy, embed_f1]
        print(scores_global)


    # Super Learner (partie 3)

    scorers = {
        "acc": accuracy_score,
        "F1": f1_score,
    }

    for scorer in scorers:
        accuracy, f1 = super_learner(tfidf_train, y2_train, tfidf_test, y2_test, scorers[scorer])
        print(accuracy, f1)

    scores_global["Super Learner"] = [None, None, None, None, accuracy, f1, None, None, None]

    df_global = pd.DataFrame.from_dict(scores_global, orient='index',
                                       columns=["bow cross-val", "bow accuracy", "bow f1",
                                                "tfidf cross-val", "tfidf accuracy", "tfidf f1",
                                                "embed cross-val", "embed accuracy", "embed f1"])

    print(df_global)
    df_global_transposed = df_global.T
    ax1 = df_global.plot.bar()
    ax2 = df_global_transposed.plot.bar()
    fig1 = ax1.get_figure()
    fig1.savefig("TP2_code/scores1.png")
    fig2 = ax2.get_figure()
    fig2.savefig("TP2_code/scores2.png")




# Entry point
if __name__ == "__main__":
    main()