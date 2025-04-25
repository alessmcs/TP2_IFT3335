# Viviane BINET (), Alessandra MANCAS (20249098)

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib
import transformers
import re  # regex library

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
def tf_idf(content):
    vectorizer = TfidfVectorizer(max_features=5000)
    count_matrix = vectorizer.fit_transform(content)
    count_array = count_matrix.toarray()
    df = pd.DataFrame(data=count_array, columns=vectorizer.get_feature_names_out())
    return df

# Entraîner modèles (Logistic regression, Random Forest, MLP)
# https://thecleverprogrammer.com/2024/02/19/compare-multiple-machine-learning-models/




def main():

    df = pd.read_csv('spam_train.csv', header=None, names=['text', 'label'])

    # Preprocess the data
    df['text'] = pretraitement(df['text'])
    # print(df.head())

    fm = bag_of_words(df['text'])
    # print(fm.head)

    tfidf = tf_idf(df['text'])
    print(tfidf.head)

    # todo: choose parameters
    models = {
        "LR": linear_model.LinearRegression(),
        "RF": RandomForestClassifier(),
        "MLP": MLPClassifier()
    }


# Entry point
if __name__ == "__main__":
    main()