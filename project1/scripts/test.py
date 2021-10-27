from implementations import *
import numpy as np

from proj1_helpers import augment

## Test of all mandatory functions with toy data set. Just to make sure they behave

# So 75 data points, 5 variables
y = np.arange(start=0, stop=75)
x = np.random.randint(low=4, high=20, size=(75, 5))
print(y)
print(x)

## Test the least square methods
max_iters = 10000
gamma = 0.001
tx = np.c_[np.ones((y.shape[0], 1)), x]
initial_w = w = np.zeros(tx.shape[1])
'''
w_GD, loss_GD = least_squares_GD(y, tx, initial_w, max_iters, gamma)
w_SGD, loss_SGD = least_squares_SGD(y, tx, initial_w, max_iters, gamma)
w_calc, loss_calc = least_squares(y, tx)

print("Losses from GD: {GD}, from SGD: {SGD} and from calculation: {calc}".format(GD=loss_GD, SGD=loss_SGD, calc=loss_calc))
print("Weights from GD: {GD}, from SGD: {SGD} and from calculation: {calc}".format(GD=w_GD, SGD=w_SGD, calc=w_calc))
print("These should be kinda the same")
print("")

for lambda_ in range(10):
    w_reg, loss_reg = ridge_regression(y, tx, lambda_)
    print("Ridge regression loss: {lo}, weights: {we} and lambda: {la}".format(lo=loss_reg, we=w_reg, la=lambda_))
print("These should increase loss and decrease weights")

'''
## Test of the logistic regression function
y = np.random.randint(low=0, high=2, size=(1,75))[0]
print(y)
lambda_ = 0
gamma = 0.01
converge_limit = 1
w_log, loss_log = reg_logistic_regression_SGD(y, tx, lambda_, initial_w, max_iters, gamma, converge_limit)

print(loss_log)

print(augment(x, [2, 3, 1, 4, 2]))
