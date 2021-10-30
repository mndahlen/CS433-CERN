# PYLIBS
import numpy as np
import matplotlib.pyplot as plt
import time

# CUSTOM MODULES
from helpers import *
from implementations import *
from feature_engineering import create_features
from train import train_model

# LOAD DATA
DATA_TRAIN_PATH = '../data/train.csv'
DATA_EVAL_PATH = '../data/test.csv'

# GET TRAIN AND TEST DATA
y_train, x_train,  idx_train = load_csv_data(DATA_TRAIN_PATH, use_pandas=True, classes=[1, -1])
_, x_eval, idx_eval = load_csv_data(DATA_EVAL_PATH, use_pandas=True, classes=[1, -1])

# CREATE FEATURES FOR TRAIN, TEST AND EVAL DATA
print("Fixing features...")
x_train, x_eval = create_features(x_train, x_eval)
x_train, y_train, x_test, y_test = split_data(x_train, y_train, 0.5, seed=7)
print("Features fixed!")

# HYPERPARAMETERS
hyperparameters = {
    "gamma": 1e-6,
    "lambda": 0,
    "max_iters": 1000,
    "initial_w": np.zeros(x_train.shape[1]),
    "conv_limit": 1e-10
}

# TRAIN
print("Training...")
w, loss = train_model(y_train, x_train, hyperparameters, model='logistic_regression', algorithm='GD')
print("Training complete!")

# EVAL TRAINED MODEL
y_eval = predict_labels(x_eval, w, classes=[1, -1])
create_csv_submission(idx_eval, y_eval, 'submission')

# TEST TRAINED MODEL
print("Testing...")
y_pred = predict_labels(x_test, w, classes=[1, -1])
accuracy = cat_accuracy(y_pred, y_train)
f1_score = F1_score(y_pred, y_train)
print("Model classified {} % correct".format(accuracy * 100))
print("Model F1-score = {}".format(f1_score))
