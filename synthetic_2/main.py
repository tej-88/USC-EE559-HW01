# Name: Tejas Acharya
# Class: EE-559 
# Date: 14-06-2023
# Assignment: Homework 1

import sys
import os
import numpy as np

CURRENT_PATH = os.getcwd()
sys.path.append(CURRENT_PATH)

from nearestMeansClassifier import NearestMeansClassifier

#DATASET FILENAME
TRAIN_DATASET_FILENAME = os.path.join(CURRENT_PATH, 'synthetic_2', 'synthetic2_train.csv')
TEST_DATASET_FILENAME = os.path.join(CURRENT_PATH, 'synthetic_2', 'synthetic2_test.csv')


def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=float)
    X = data[:, :-1]
    y = data[:, -1].astype('int32')
    return (X, y)

#LOAD DATA
X_train, y_train = load_data(TRAIN_DATASET_FILENAME)
X_test, y_test = load_data(TEST_DATASET_FILENAME)

features_idx = np.array([0, 1])

#NEAREST MEANS CLASSIFIER MODEL
model = NearestMeansClassifier()

#Fit the Model
model.fit(X_train, y_train, features_idx)

#Predict the TRAIN DATA
y_train_predict = model.predict(X_train)

#Error Rate on TRAIN DATA
train_error_rate = model.get_error_rate(y_train, y_train_predict)

#Plot the Boundary
model.plot_boundary(X_train, y_train)
model.plot_boundary(X_train, y_train_predict)

#Predict the TEST DATA
y_test_predict = model.predict(X_test)

#Error Rate on TEST DATA
test_error_rate = model.get_error_rate(y_test, y_test_predict)

print(f'Train Error Rate: {train_error_rate:.2f}%')
print(f'Test Error Rate: {test_error_rate:.2f}%')