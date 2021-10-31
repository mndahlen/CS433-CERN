# PYLIBS
import numpy as np
import matplotlib.pyplot as plt

# CUSTOM MODULES
from helpers import *
from implementations import *
from feature_engineering import create_features
from train import train_model
from algorithms import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-folds cross validation."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

# LOAD DATA
DATA_TRAIN_PATH = '../data/train.csv'

# GET DATA
y_data, x_data,  idx_data = load_csv_data(DATA_TRAIN_PATH, use_pandas=True, classes=[1, -1])

# CREATE FEATURES FOR EACH 
feature_sets_for_validation = [[[0]],[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],
                               [[7]],[[8]],[[9]],[[10]],[[11]],[[12]],[[13]],
                               [[14]],[[15]],[[16]],[[17]],[[18]],[[19]],
                               [[20]],[[21]],[[22]],[[23]],[[24]],[[25]],
                               [[26]],[[27]],[[28]],[[29]]]

# HYPERPARAMETERS
hyperparameters = {
    "gamma": 1e-7,
    "lambda": 0.6,
    "max_iters": 100,
    "initial_w": np.zeros(len(feature_sets_for_validation[0])),
    "conv_limit": 1e-10
}

# INIT MODEL
model = build_model("logistic_regression","LSAGDR",hyperparameters)

# SET PARAMS FOR CROSS VAL
k_fold = 2
k_indices = build_k_indices(y_data,k_fold,5)

## RUN CROSS VAL ACROSS FEATURES
all_tests = []
y = y_data
for k in range(1,k_fold+1):
    k_test = []
    for feature_set in feature_sets_for_validation:
        print("K = {}, feature set = {}".format(k,feature_set))
        x, x_ = create_features(x_data, x_data,feature_set)
        x_test = x[k_indices[k-1]]
        y_test = y[k_indices[k-1]].flatten()
        
        x_train = np.delete(x,k_indices[k-1],axis=0)
        y_train = np.delete(y,k_indices[k-1],axis=0).flatten()

        w, loss = model(y_train, x_train)
        y_pred = predict_labels(x_test, w, classes=[1, -1])
        k_test.append(cat_accuracy(y_pred, y_test))

    all_tests.append(np.asarray(k_test))

all_test_np = np.asarray(all_tests)

# GET MEANS
all_test_mean = np.mean(all_test_np, axis=0)

# PLOT RESULT
labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
plt.bar(labels, list(all_test_mean))
plt.xlabel("feature index")
plt.ylabel("accuracy [%]")
plt.title("cross validation")
plt.grid(True)
plt.show()


