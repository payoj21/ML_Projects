import os
import json
from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

SEED = 5

stop_words = text.ENGLISH_STOP_WORDS
pos_words = [('insightful',0.6),('clever',0.5),('comical',0.5),('charismatic',0.5),('enjoyable',0.6),('uproarious',0.8),('original',0.7),('tender',0.5),
            ('hilarious',0.6),('absorbing',0.7),('sensitive',0.6),('riveting',0.6),('intriguing',0.5),('fascinating',0.4),('pleasant',0.3),
             ('dazzling',0.6),('thought provoking',0.7),('imaginative',0.6),('legendary',0.7),('unpretentious',0.4)]

neg_words = [('violent',0.4),('moronic',0.7),('flawed',0.5),('juvenile',0.6),('boring',0.5),('distasteful',0.6),('disgusting',0.8),('senseless',0.6),
             ('brutal',0.3),('confused',0.3),('disappointing',0.7),('silly',0.4),('predictable',0.5),('stupid',0.4),('uninteresting',0.5),
             ('incredibly tiresome',0.8),('cliche ridden',0.5),('outdated',0.5),('dreadful',0.7),('bland',0.5)]



'''
The ItemSelector class was created by Matt Terry to help with using
Feature Unions on Heterogeneous Data Sources

All credit goes to Matt Terry for the ItemSelector class below

For more information:
http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
'''
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


"""
This is an example of a custom feature transformer. The constructor is usedss
to store the state (e.g like if you need to store certain words/vocab), the
fit method is used to update the state based on the training data, and the
transform method is used to transform the data into the new feature(s). In
this example, we simply use the length of the movie review as a feature. This
requires no state, so the constructor and fit method do nothing.
"""
class TextLengthTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, examples):
        return self

    def transform(self, examples):
        features = np.zeros((len(examples), 1))
        i = 0
        for ex in examples:
            features[i, 0] = len(ex)
            i += 1

        return features

# TODO: Add custom feature transformers for the movie review data

class word_connotation(BaseEstimator, TransformerMixin):
    def _init_(self):
        pass
    def fit(self, examples):
        return self
    def transform(self, examples):
        features = np.zeros((len(examples),1))
        i=0
        for ex in examples:

            for word in pos_words:
                if word[0] in ex:
                    features[i,0] += word[1]
            for word in neg_words:
                if word[0] in ex:
                    features[i,0] -= word[1]
            i += 1
        return features

class punctuations(BaseEstimator, TransformerMixin):
    def _init_(self):
        pass
    def fit(self, examples):
        return self
    def transform(self, examples):
        features = np.zeros((len(examples),1))
        i=0
        for ex in examples:
            count = 0
            if('!' in ex):
                count = ex.count('!')

                if(count > 1):
                    features[i,0] = count
            i += 1
        return features
class question_marks(BaseEstimator, TransformerMixin):
    def _init_(self):
        pass
    def fit(self, examples):
        return self
    def transform(self, examples):
        features = np.zeros((len(examples),1))
        i=0
        for ex in examples:
            count = 0
            if('?' in ex):
                count = ex.count('?')
                if(count > 1):
                    features[i,0] = count
            i += 1
        return features
class num_of_sentences(BaseEstimator, TransformerMixin):
    def _init_(self):
        pass
    def fit(self, examples):
        return self
    def transform(self, examples):
        features = np.zeros((len(examples),1))
        i=0
        for ex in examples:
            count = ex.count('.')
            features[i,0] = count
            i += 1
        return features
class Featurizer:
    def __init__(self):
        # To add new features, just add a new pipeline to the feature union
        # The ItemSelector is used to select certain pieces of the input data
        # In this case, we are selecting the plaintext of the input data

        # TODO: Add any new feature transformers or other features to the FeatureUnion
        self.all_features = FeatureUnion([
            ('text_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('text_length', TextLengthTransformer()),

            ])  ),
            ('word_stats', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('tfidf', TfidfVectorizer(stop_words= stop_words))
             ])),
            ('n_grams', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('ngram_count', CountVectorizer(ngram_range=(1, 3))),
            ])),
            ('word_connotation', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('review_connotation_score', word_connotation()),

            ])),
            ('punctuations', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('exclamation', punctuations()),
            ])),
            ('question',Pipeline([
                ('selector', ItemSelector(key='text')),
                ('question_marks', question_marks()),
            ])),
            ('sentences', Pipeline([
                ('selector', ItemSelector(key='text')),
                ('num_of_sentences', num_of_sentences()),
            ]))
        ])

    def train_feature(self, examples):
        return self.all_features.fit_transform(examples)

    def test_feature(self, examples):
        return self.all_features.transform(examples)

if __name__ == "__main__":

    # Read in data

    dataset_x = []
    dataset_y = []

    with open('../data/movie_review_data.json') as f:
        data = json.load(f)
        for d in data['data']:
            dataset_x.append(d['text'])
            dataset_y.append(d['label'])

    # Split dataset
    # X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.3, random_state=SEED)
    #
    #
    # feat = Featurizer()
    #
    # labels = []
    # for l in y_train:
    #     if not l in labels:
    #         labels.append(l)
    #
    # print("Label set: %s\n" % str(labels))
    #
    # # Here we collect the train features
    # # The inner dictionary contains certain pieces of the input data that we
    # # would like to be able to select with the ItemSelector
    # # The text key refers to the plaintext
    # feat_train = feat.train_feature({
    #     'text': [t for t in X_train]
    # })
    # # Here we collect the test features
    # feat_test = feat.test_feature({
    #     'text': [t for t in X_test]
    # })
    #
    # #print(feat_train)
    # #print(set(y_train))
    #
    # # Train classifier
    # lr = SGDClassifier(loss='log', penalty='l2', alpha=0.0025, max_iter=7000, shuffle=True, verbose=2)
    #
    # lr.fit(feat_train, y_train)
    # y_pred = lr.predict(feat_train)
    # accuracy = accuracy_score(y_pred, y_train)
    # print("Accuracy on training set =", accuracy)
    # y_pred = lr.predict(feat_test)
    # accuracy = accuracy_score(y_pred, y_test)
    # print("Accuracy on test set =", accuracy)

    # EXTRA CREDIT: Replace the following code with scikit-learn cross validation
    # and determine the best 'alpha' parameter for regularization in the SGDClassifier

    alphas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    average_accuracy = []
    for a in alphas:

        kf = KFold(n_splits=5)
        test_accuracy = []
        count = 0
        for train_index, test_index in kf.split(dataset_x):
            count +=1
            new_X_train = []
            new_y_train = []
            new_X_test = []
            new_y_test = []
            for index in train_index:
                new_X_train.append(dataset_x[index])
                new_y_train.append(dataset_y[index])
            for index in test_index:
                new_X_test.append(dataset_x[index])
                new_y_test.append(dataset_y[index])



            feat = Featurizer()


            feat_train = feat.train_feature({
                'text': [t for t in new_X_train]

            })
            feat_test = feat.test_feature({
                'text': [t for t in new_X_test]

            })
            lr = SGDClassifier(loss='log', penalty='l2', alpha=a, max_iter=10000, shuffle=True, verbose=0)

            lr.fit(feat_train, new_y_train)
            y_pred = lr.predict(feat_train)
            accuracy = accuracy_score(y_pred, new_y_train)
            print("Accuracy on training set =", accuracy)
            y_pred = lr.predict(feat_test)
            accuracy = accuracy_score(y_pred, new_y_test)
            print("Accuracy on test set =", accuracy, count)
            test_accuracy.append(accuracy)

        test_accuracy_avg = sum(test_accuracy)/len(test_accuracy)
        print('K fold for alpha = ', a)
        print('Average Accuracy: ',test_accuracy_avg)
        average_accuracy.append((a,test_accuracy_avg))
    print(average_accuracy)

    plt.plot(alphas, average_accuracy, "ro")

    plt.suptitle("Alphas vs Accuracy Plot")
    plt.xlabel("Alphas")
    plt.ylabel("Accuracy")
    plt.show()
