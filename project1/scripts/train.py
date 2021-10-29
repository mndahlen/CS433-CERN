from implementations import reg_logistic_regression, logistic_regression
import numpy as np

def train_model(y_train, x_train, hyperparameters, algorithm = "logistic_regression", multimodel = False):

    if algorithm == "logistic_regression":
        loss = np.NaN
        gamma = hyperparameters["gamma"]
        initial_w = hyperparameters["initial_w"]
        max_iters = hyperparameters["max_iters"]
        conv_limit = hyperparameters["conv_limit"]
        w, loss = logistic_regression(y_train, x_train, initial_w, max_iters, gamma, conv_limit)
    elif algorithm == "reg_logistic_regression":
        loss = np.NaN
        gamma = hyperparameters["gamma"]
        initial_w = hyperparameters["initial_w"]
        max_iters = hyperparameters["max_iters"]
        conv_limit = hyperparameters["conv_limit"]
        lambda_ = hyperparameters["lambda"]
        w, loss = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)
    else:
        print("Invalid train algorithm selected")

    return w, loss