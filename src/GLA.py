"""
GDA

"""

# Author: Jasper John Jaso <jasperjohnjaso6969@gmail.com>
# Created: October 4, 2024
# Description: A Python program that implements a Naive Bayes Classifier from scratch.

import numpy as np

class NaiveBayesClassifier:
    """
    This is a docstring for NaiveBayesClassifier.
    
    Naive Bayes Classifier is a probabilistic machine 
    learning model based on Bayes' Theorem, assuming 
    that the features are conditionally independent 
    given the class label. It's efficient for classification 
    tasks and is often used for text classification
    """

    def __init__(self, alpha=1.0):
        self.Phi_jy = None
        self.Phi_y = None
        self.n_label = None
        self.alpha=alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.n_labels = len(np.unique_counts(y))
        self.Phi_jy = np.zeros((self.n_labels, n_features))
        self.Phi_y = np.zeros(self.n_labels)
        
        for label in range(self.n_labels):

            self.Phi_jy[label] = self.Get_prob_per_word(X[y == label], y[y == label])

        self.Phi_y[1] = self.Get_class_prior(y[y == 1], y)
        self.Phi_y[0] = 1 - self.Phi_y[1]

    def predict(self, X):
        # X_con = np.array(X)
        X_con = np.where(X != 0, 1, 0)
        i, j = X_con.shape
        prediction = np.zeros(i)
        predict_con = np.zeros((self.n_labels, i))

        for label in range(self.n_labels):
            
            for sample in range(i):
                X_i = X_con[sample]
                nom = self.Phi_y[label] * np.prod(self.Phi_jy[label][X_i == 1])
                dom = (self.Phi_y[1] * np.prod(self.Phi_jy[1][X_i == 1])) + (self.Phi_y[0] * np.prod(self.Phi_jy[0][X_i == 1]))
                predict_con[label][sample]  = nom / dom
            
        prediction = np.argmax(predict_con, axis=0)

        return prediction

    """
    Method to get the Parameters
    """

    def Get_prob_per_word(self, X, y):
        X_con = np.where(X != 0, 1, 0)
        i, j = X.shape
        phi_con = np.zeros(j)

        for feature in range(j):
            
            phi_con[feature] = (np.sum((X_con[X_con.T[feature] == 1])) + self.alpha) / (len(y) + (1 + self.alpha))
           
        return phi_con
    
    def Get_class_prior(self, y_1, y):
        return len(y_1) / len(y)
