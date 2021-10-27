# Try to see how each variable is doing in different augmentation
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.core.numeric import NaN

####### LOAD DATA
from proj1_helpers import *
#from logistic_implementations import logistic_gradient_descent, stoc_logistic_gradient_descent, calculate_logistic_loss
from implementations import build_poly_1D, reg_logistic_regression, split_data, logistic_regression, calculate_logistic_loss
DATA_TRAIN_PATH = '/home/arvid/Documents/Machine-Learning/CS433/project1/data/train.csv'
DATA_TEST_PATH = '/home/arvid/Documents/Machine-Learning/CS433/project1/data/test.csv'

y, x,  idx = load_csv_data(DATA_TRAIN_PATH)

#########
x, mean_x, std_x = standardize(x)

max_aug = 4
x = np.array(x)
te_losses = []
for var in range(len(x[0])):
    x_simple = x[:, var]
    var_losses = []
    for exp in range(1, max_aug + 1):
        x_aug = build_poly_1D(x_simple, exp)
        x_aug,y_train,x_test,y_test = split_data(x_aug, y, 0.5, seed=7)
        init_w = np.zeros(len(x_aug[0]))
        max_iters = 50000
        gamma = 1e-5
        conv_limit = 1e-5
        loss = np.nan
        while(np.isnan(loss)):
            print("Started reg loop")
            w, loss = logistic_regression(y_train, x_aug, init_w, max_iters, gamma, conv_limit)
            gamma = gamma/10
        te_loss = calculate_logistic_loss(y_test, x_test, np.array(w))
        y_pred = predict_labels(w, x_test)

        correct = 0
        for i, yi in enumerate(y_pred):
            if ((yi == y_test[i]) or (yi == -1 and y_test[i] == 0)):
                correct += 1
        te_loss = correct/len(y_pred)
        var_losses.append(te_loss)
    te_losses.append(var_losses)

print(te_losses)

fields = ['degree 1', 'degree 2','degree3', 'degree 4']

with open('each_var_test.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(fields)
    write.writerows(te_losses)