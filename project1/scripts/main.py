import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.core.function_base import linspace
from numpy.core.numeric import NaN

# LOAD DATA
from helpers import *
#from logistic_implementations import logistic_gradient_descent, stoc_logistic_gradient_descent, calculate_logistic_loss
from implementations import reg_logistic_regression, split_data, logistic_regression
from train import train_model
DATA_TRAIN_PATH = '../data/train.csv'
DATA_EVAL_PATH = '../data/test.csv'

# GET TRAIN AND TEST DATA
#y_train, x_train,  idx_train = load_csv_data(DATA_TRAIN_PATH)
#yb, x_eval, ids_eval = load_csv_data(DATA_EVAL_PATH)
y_train, x_train,  idx_train = load_csv_data(DATA_TRAIN_PATH, use_pandas=True, classes=[1,0])
yb, x_eval, ids_eval = load_csv_data(DATA_EVAL_PATH, use_pandas=True, classes=[1,0])
#print(y_train.shape)
#print(y_train_pd.shape)
#exit
# CREATE FEATURES FOR TRAIN, TEST AND EVAL DATA
print("Fixing features...")
x_train, x_eval = create_features(x_train, x_eval)
x_train,y_train,x_test,y_test = split_data(x_train, y_train, 0.5, seed=7)
print("Features fixed!")

# HYPERPARAMETERS
hyperparameters = {
    "gamma":1e-4,
    "lambda":0,
    "max_iters" : 1000,
    "initial_w" : np.zeros(len(x_train[0])),
    "conv_limit" : 1e-10
}

# TRAIN
print("Training...")
w, loss = train_model(y_train, x_train, hyperparameters, algorithm = "logistic_regression")
print("Training complete!")

# TEST TRAINED MODEL
print("Testing...")
w = w.reshape(17,1)
y_pred = predict_labels(x_test, w, classes = [1,0])
diff = y_pred - y_test.reshape(y_test.shape[0],1)
print("Model classified {} % correct".format((1 -  sum(abs(diff))/y_test.shape[0])[0]))

# EVAL TRAINED MODEL
y_eval = predict_labels(x_eval, w, classes = [1,-1])
create_csv_submission(ids_eval, y_eval, 'submission')



