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
DATA_TEST_PATH = '../data/test.csv'

# GET TRAIN AND TEST DATA
t0 = time.time()
print("Loading data with really slow given function instead of fast Pandas...")
y_train, x_train,  idx_train = load_csv_data(DATA_TRAIN_PATH)
yb, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
print("Finally completed loading data!\nThat was {} seconds! Pandas does the same in 6 seconds zipped!".format(int(time.time() - t0)))

# CREATE FEATURES FOR TRAINING
print("Fixing features...")
x_train, x_test = create_features(x_train, tX_test)
x_train,y_train,my_x_test,my_y_test = split_data(x_train, y_train, 0.5, seed=7)
print("Features fixed!")

# HYPERPARAMETERS
hyperparameters = {
    "gamma":1e-4,
    "lambda":0,
    "max_iters" : 100000,
    "initial_w" : np.zeros(len(x_train[0])),
    "conv_limit" : 1e-10
}

# TRAIN
print("Training...")
w,loss = train_model(y_train, x_train, hyperparameters, algorithm = "logistic_regression")
print("Training complete!")

# TEST TRAINED MODEL
print("Testing...")
print(final_w)
my_y_pred = predict_labels(final_w, my_x_test)
print("Testing complete!")

correct = 0
for i, y in enumerate(my_y_pred):
    if ((y == my_y_test[i]) or (y == -1 and my_y_test[i] == 0)):
        correct += 1
print("Correct percentage on test set: {co}".format(co=correct/len(my_y_pred)))

y_pred = predict_labels(final_w, x_test)

create_csv_submission(ids_test, y_pred, 'Arvid_submission')



