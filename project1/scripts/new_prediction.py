import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from numpy.core.numeric import NaN

####### LOAD DATA
from proj1_helpers import *
from logistic_implementations import logistic_gradient_descent, stoc_logistic_gradient_descent, calculate_logistic_loss
from implementations import reg_logistic_regression, split_data, logistic_regression
DATA_TRAIN_PATH = '/home/arvid/Documents/Machine-Learning/CS433/project1/data/train.csv'
DATA_TEST_PATH = '/home/arvid/Documents/Machine-Learning/CS433/project1/data/test.csv'

y_train, x_train,  idx_train = load_csv_data(DATA_TRAIN_PATH)
## Get test data
yb, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

print(x_train[0])

#########

## Fix with the train data. Remove some features, augment some other. Do the same with test matrix
remove_vars = [2, 4, 7, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

augment_vars = [3, 1, 1, 1, 2, 1, 4, 1]
x_train, mean_x, std_x = standardize(x_train)
x_test, mean_x_test, std_x_test = standardize(tX_test)
x_train = remove_variables(x_train, remove_vars)
x_test = remove_variables(x_test, remove_vars)
x_train = augment(x_train, augment_vars)
x_test = augment(x_test, augment_vars)

x_train,y_train,my_x_test,my_y_test = split_data(x_train, y_train, 0.5, seed=7)
x_train = np.array(x_train)
## 

## Now, do the training

## Got these weights with this function, can maybe run these through regulazied?
w_submitted = [ 1.84218317e+01, -1.07724799e+01, -8.28169071e+00, -6.65540347e-01, 
1.33666649e-01, -1.32453800e-02,  1.00520283e+00,  4.90903200e+00,
 -6.24885280e+00, -1.86601977e-01,  8.27258145e-02, -5.45346629e-03,
  7.24767877e-02,  4.62745843e-01,  1.81722650e-02,  4.65404928e-01,
 -5.52326796e-02]

w_regula = [11.16703908, -6.08146579, -4.80851339, -0.63012265, -0.04354135,  1.08390525,
  2.96881392, -4.21631327, -0.10367624,  0.23483634,  0.46485023, -0.07612583,
 -0.01322689,  0.37786607]
gamma = 1e-6
_lambda = 0
max_iters = 10000
#initial_w = np.zeros(len(x_train[0]))
#initial_w = np.array(w_submitted)
initial_w = np.array(w_regula)
final_w = initial_w
loss = np.nan
conv_limit = 1e-8
while (np.isnan(loss)):
    print("Started loop")
    w, loss = reg_logistic_regression(y_train, x_train, initial_w, max_iters, gamma, conv_limit, _lambda)
    gamma = gamma/10
    final_w = w


# Finished with training
print(final_w)
my_y_pred = predict_labels(final_w, my_x_test)

correct = 0
for i, y in enumerate(my_y_pred):
    if ((y == my_y_test[i]) or (y == -1 and my_y_test[i] == 0)):
        correct += 1
print("Correct percentage on test set: {co}".format(co=correct/len(my_y_pred)))

y_pred = predict_labels(final_w, x_test)

create_csv_submission(ids_test, y_pred, 'Arvid_submission')


