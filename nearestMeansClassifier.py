# Name: Tejas Acharya
# Class: EE-559 
# Date: 14-06-2023
# Assignment: Homework 1

import numpy as np
from plotDecBoundaries import plotDecBoundaries

class NearestMeansClassifier():
    def __init__(self):
        self.C = 0
        self.means = None
        self.classes = None
        self.features_idx = None

    
    def fit(self, X, y, features_idx):
        self.classes = np.unique(y)
        self.C = len(self.classes)
        self.features_idx = features_idx
        D = len(features_idx)

        self.means = np.empty((self.C, D))

        total_sum = np.zeros((self.C, D))
        N = np.zeros((self.C, ))

        for i in range(len(y)):
            total_sum[y[i] - 1, :] += X[i, features_idx]
            N[y[i] - 1] += 1
        
        for j in range(self.C):
            self.means[j, :] = total_sum[j,:] / N[j]
        
        return


    def predict(self, X):
        N = len(X)

        y_hat = np.empty((N,))

        for i in range(N):
            y_hat[i] = self.get_nearest_class(X[i, self.features_idx])
        
        y_hat = y_hat.astype('int32')

        return y_hat


    def get_nearest_class(self, x):
        l2_distances = np.empty((self.C, ))

        for i in range(self.C):
            l2_distances[i] = self.get_l2_norm(self.means[i, :], x)
        
        return self.classes[np.argmin(l2_distances)]

    
    def get_l2_norm(self, a, b):
        e = a - b
        return np.sqrt(np.dot(e.T, e))

    
    def get_error_rate(self, y, y_hat):
        return (sum(y != y_hat) / len(y)) * 100

    
    def get_class_means(self):
        return self.means