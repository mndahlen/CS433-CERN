import numpy as np


# Functions to implement
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient_LS(y, tx, w)
        # Update Weights with Gradient Descent
        w = w - gamma * gradient
    loss = compute_mse(y, tx, w)  # are we sure that loss is mse ?
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    w = initial_w
    for n_iter in range(max_iters):
        idx = np.random.randint(len(y))
        gradient = compute_stochastic_gradient_LS(y, tx, w, idx)
        # Update Weights with Stochastic Gradient Descent
        w = w - gamma * gradient
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution with normal equations."""
    w = np.linalg.solve(np.matmul(np.transpose(tx), tx), np.matmul(np.transpose(tx), y))
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Calculate the ridge regression solution with normal equations."""
    n = y.shape[0]
    lambda_prime = 2 * n * lambda_
    w = np.linalg.solve(np.matmul(np.transpose(tx), tx) + lambda_prime * np.eye(tx.shape[1]), np.matmul(np.transpose(tx), y))
    loss = compute_rmse(y, tx, w)  # which loss to choose ?
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for logistic regression."""

    w = initial_w
    for iter_ in range(max_iters):
        gradient = compute_gradient_log_reg(y, tx, w)
        w = w - gamma * gradient
    loss = compute_neg_log_likelihood(y, tx, w)
    return w, loss


def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm for logistic regression."""

    w = initial_w
    for iter_ in range(max_iters):
        idx = np.random.randint(len(y))
        gradient = compute_stochastic_gradient_log_reg(y, tx, w, idx)
        w = w - gamma * gradient
    loss = compute_neg_log_likelihood(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Gradient descent algorithm for regularized logistic regression."""

    w = initial_w
    for iter_ in range(max_iters):
        gradient = compute_gradient_log_reg_l2(y, tx, w, lambda_)
        w = w - gamma * gradient
    loss = compute_neg_log_likelihood_l2(y, tx, w, lambda_)
    return w, loss


def reg_stochastic_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm for regularized logistic regression."""

    w = initial_w
    for iter_ in range(max_iters):
        idx = np.random.randint(len(y))
        gradient = compute_stochastic_gradient_log_reg_l2(y, tx, w, lambda_, idx)
        w = w - gamma * gradient
    loss = compute_neg_log_likelihood_l2(y, tx, w, lambda_)
    return w, loss


# Compute gradients
def compute_gradient_LS(y, tx, w):
    """Compute the gradient for least squares cost function."""
    e = y - np.matmul(tx, w)
    n = y.shape[0]
    return (-1 / n) * np.matmul(np.transpose(tx), e)


def compute_stochastic_gradient_LS(y, tx, w, i):
    """Compute the stochastic gradient for least squares cost function."""
    n = y.shape[0]
    return np.expand_dims((-1 / n) * (np.dot(np.transpose(tx[i]), w) - y[i]) * np.transpose(tx[i]), 1)


def compute_gradient_log_reg(y, tx, w):
    """Compute the gradient of loss neg log-likelihood."""
    return np.matmul(np.transpose(tx), np.multiply(- y, sigmoid(np.multiply(-y, np.matmul(tx, w)))))


def compute_stochastic_gradient_log_reg(y, tx, w, i):
    """Compute the stochastic gradient of loss neg log-likelihood."""
    return np.expand_dims(- y[i] * sigmoid(-y[i] * np.dot(tx[i], w)) * np.transpose(tx[i]), 1)


def compute_gradient_log_reg_l2(y, tx, w, mu):
    """Compute the gradient of regularized loss neg log-likelihood."""
    return np.matmul(np.transpose(tx), np.multiply(- y, sigmoid(np.multiply(-y, np.matmul(tx, w))))) + mu * w


def compute_stochastic_gradient_log_reg_l2(y, tx, w, mu, i):
    """Compute the stochastic gradient of regularized loss neg log-likelihood."""
    return np.expand_dims(- y[i] * sigmoid(-y[i] * np.dot(tx[i], w)) * np.transpose(tx[i]), 1) + mu * w


# According to me these 2 next functions should be elsewhere
def build_poly_1D(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.c_[np.ones(x.shape[0]),x]
    for d in range(2,degree + 1):
        poly = np.c_[poly, np.power(x,d)]
    return poly


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # Set seed and shuffle data
    np.random.seed(seed)
    x_w_output = np.c_[x,y]
    np.random.shuffle(x_w_output)
    x_shuffled = x_w_output[:,:-1]
    y_shuffled = x_w_output[:,-1]
    # Prepare split index
    split_index = int(np.round((x_shuffled.shape)[0]*(ratio)))

    # Split
    x_train = x_shuffled[:split_index]
    y_train = y_shuffled[:split_index]
    x_test = x_shuffled[split_index:]
    y_test = y_shuffled[split_index:]

    return x_train,y_train,x_test,y_test


# Cost functions
def compute_rmse(y, tx, w):
    mse = compute_mse(y, tx, w)
    return np.sqrt(2 * mse)


def compute_mse(y, tx, w):
    e = y - np.matmul(tx, w)
    n = y.shape[0]
    return (1 / (2 * n)) * np.linalg.norm(e)**2


def compute_mae(y, tx, w):
    e = y - np.matmul(tx, w)
    n = y.shape[0]
    return (1 / n) * np.sum(np.abs(e))


def compute_neg_log_likelihood(y, tx, w):
    """Compute negative log likelihood."""
    return np.sum(np.log(1 + np.exp(np.multiply(-y, np.matmul(tx, w)))))


def compute_neg_log_likelihood_l2(y, tx, w, mu):
    """Compute negative log likelihood."""
    return np.sum(np.log(1 + np.exp(np.multiply(-y, np.matmul(tx, w))))) + (mu / 2) * np.linalg.norm(w)**2


# For logistic regression
def sigmoid(t):
    """Apply the sigmoid function on t."""
    return 1 / (1 + np.exp(-t))




