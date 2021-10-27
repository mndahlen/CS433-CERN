# Useful starting lines

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from proj1_helpers import *
from implementations import *

## Visualize all variables contra y. 
def visual_all_vars(y, x):
    # Should return mean and std of each variable in x, or the standardized
    x, mean_x, std_x = standardize(x)

    num_of_var = len(x[0])
    rows = int(np.floor(np.sqrt(num_of_var)))
    cols = int(num_of_var/rows)
    fig, axs = plt.subplots(rows, cols)

    for row in range(rows):
        for col in range(cols):
            axs[row, col].plot(x[:, row*rows + col], y)
    plt.show()

## Histogram plots of all variables. Plots for y = 0 and y = 1
def hists(y, x):

    # 
    x_0 = [x[i] for i in range(len(x)) if not y[i]]
    x_1 = [x[i] for i in range(len(x)) if y[i]]
    
    num_of_var = len(x[0])
    rows = int(np.floor(np.sqrt(num_of_var)))
    cols = int(num_of_var/rows)
    fig, axs = plt.subplots(rows, cols)

    for row in range(rows):
        for col in range(cols):

            # Plot, and try to remove outliers (-999 are present kinda everywhere and
            # not close to anythin else)
            x_0_tmp = [k[row*rows + col] for k in x_0 if k[row*rows + col] != -999]
            x_1_tmp = [k[row*rows + col] for k in x_1 if k[row*rows + col] != -999]
            min_val = min([min(x_0_tmp), min(x_1_tmp)])
            max_val = max([max(x_0_tmp), max(x_1_tmp)])
            ra = (min_val, max_val)
            #print(ra)


            axs[row, col].hist(x_0_tmp, bins=100, range=ra, alpha=0.5)
            axs[row, col].hist(x_1_tmp, bins=100, range=ra, alpha=0.5)
    plt.show()
from proj1_helpers import *
from implementations import *

DATA_TRAIN_PATH = '~/Documents/Machine-Learning/CS433/project1/data/train.csv.zip'
DATA_TEST_PATH = '~/Documents/Machine-Learning/CS433/project1/data/test.csv.zip'

# Load data
data_train = pd.read_csv(DATA_TRAIN_PATH)
data_test = pd.read_csv(DATA_TEST_PATH) # NOTE: This is test for SUBMISSION, not test for training.

# Make Y binary [-1,1]
data_train.loc[data_train['Prediction'] == 's','Prediction'] = 1
data_train.loc[data_train['Prediction'] == 'b','Prediction'] = 0
data_train['Prediction'] = pd.to_numeric(data_train['Prediction'])

# Extract y
y_train = data_train['Prediction'].to_frame()
y = y_train.to_numpy()

# Get ids
id_train = data_train['Id'].to_frame()

# Get x
x_train = data_train.copy()
x_train.drop(['Id','Prediction'], 1, inplace=True) 
x = x_train.to_numpy()


#visual_all_vars(y, x)
hists(y, x)
