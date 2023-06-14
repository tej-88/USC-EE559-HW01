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
TRAIN_DATASET_FILENAME = os.path.join(CURRENT_PATH, 'synthetic_1', 'synthetic1_train.csv')
TEST_DATASET_FILENAME = os.path.join(CURRENT_PATH, 'synthetic_1', 'synthetic1_test.csv')
