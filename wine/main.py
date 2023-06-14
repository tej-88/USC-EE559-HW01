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
TRAIN_DATASET_FILENAME = os.path.join(CURRENT_PATH, 'wine', 'wine_train.csv')
TEST_DATASET_FILENAME = os.path.join(CURRENT_PATH, 'wine', 'wine_test.csv')


def load_data(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=float)
    X = data[:, :-1]
    y = data[:, -1].astype('int32')
    return (X, y)


#LOAD DATA
X_train, y_train = load_data(TRAIN_DATASET_FILENAME)
X_test, y_test = load_data(TEST_DATASET_FILENAME)

num_features = X_test.shape[1]

#(c) Features - [alcohol content, malic acid content]
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


#(d) Best Feature
features_list = []
for i in range(num_features):
    for j in range(num_features):
        feature_set = {i, j}
        if feature_set not in features_list:
            features_list.append(feature_set)

train_error_rate_list = []

for feature in features_list:
    model = NearestMeansClassifier()
    model.fit(X_train, y_train, list(feature))
    y_train_predict = model.predict(X_train)
    train_error_rate = model.get_error_rate(y_train, y_train_predict)
    train_error_rate_list.append(train_error_rate)

train_error_rate_list = np.array(train_error_rate_list)

best_feature = list(features_list[np.argmin(train_error_rate_list)])
print(f'Best Feature: {best_feature}')

best_model = NearestMeansClassifier()
best_model.fit(X_train, y_train, best_feature)
y_train_predict_best = best_model.predict(X_train)
train_error_rate_best = model.get_error_rate(y_train, y_train_predict_best)
best_model.plot_boundary(X_train, y_train)
best_model.plot_boundary(X_train, y_train_predict_best)

y_test_predict_best = best_model.predict(X_test)
test_error_rate_best = model.get_error_rate(y_test, y_test_predict_best)

print(f'Train Error Rate(Best Feature): {train_error_rate_best:.2f}%')
print(f'Test Error Rate(Best Feature): {test_error_rate_best:.2f}%')


#(e)
test_error_rate_list = []

for feature in features_list:
    model = NearestMeansClassifier()
    model.fit(X_train, y_train, list(feature))
    y_test_predict = model.predict(X_test)
    test_error_rate = model.get_error_rate(y_test, y_test_predict)
    test_error_rate_list.append(test_error_rate)

test_error_rate_list = np.array(test_error_rate_list)

train_error_mean = train_error_rate_list.mean()
test_error_mean = test_error_rate_list.mean()


train_error_std = train_error_rate_list.std()
test_error_std = test_error_rate_list.std()

train_error_cv = train_error_std / train_error_mean
test_error_cv = test_error_std / test_error_mean

print(f'Train Error STD: {train_error_std:.2f}')
print(f'Train Error CV: {train_error_cv:.2f}')

print(f'Test Error STD: {test_error_std:.2f}')
print(f'Test Error CV: {test_error_cv:.2f}')