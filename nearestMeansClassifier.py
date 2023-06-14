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