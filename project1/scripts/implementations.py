import numpy as np
import random
from helpers import *

from numpy.core.numeric import Inf

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    w = initial_w
    for n_iter in range(max_iters):
        G = compute_gradient(y,tx,w)
        w = w - gamma*G
    loss = compute_mse(y, tx, w)
    return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    w = initial_w
    for n_iter in range(max_iters):
        idx = np.random.randint(len(y))
        e = y[idx] - tx[idx].dot(w)
        Stoch_G = -tx[idx]*e
        w = w - gamma*Stoch_G
    loss = compute_mse(y, tx, w)
    return (w, loss)

def least_squares(y, tx):
    """calculate the least squares solution."""

    w = np.linalg.solve(np.transpose(tx)@tx, np.transpose(tx)@y)
    loss = compute_mse(y, tx, w)
    return (w, loss)

def ridge_regression(y, tx, lambda_):
    """ridge regression."""
    N = len(y)
    lambda_prime = 2*N*lambda_
    w = np.linalg.solve(np.transpose(tx)@tx + lambda_prime*np.eye(tx.shape[1]),np.transpose(tx)@y)
    loss = compute_rmse(y,tx,w)
    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma, converge_limit):
    '''Logistic regression with gradient descent'''

    w = initial_w
    last_loss = 0
    loss = Inf
    for iter_ in range(max_iters):
        if iter_%500 == 0:
            last_loss = loss
            loss = calculate_logistic_loss(y, tx, w)
            print("iter: {}/{}, loss = {}\n".format(iter_,max_iters,loss))
                        
        grad = calculate_logistic_gradient(y, tx, w)
        w = w - gamma*grad
        
    loss = calculate_logistic_loss(y, tx, w)
    print("iter: {}/{}, loss = {}\n".format(iter_,max_iters,loss))
    return (w, loss)

def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma,batch_ratio = 0.5,):
    '''Logistic regression with stochastic gradient descent'''

    w = initial_w
    size_data = y.shape[0]
    batch_size = int(size_data*batch_ratio)
    for iter_ in range(max_iters):
        if iter_%1000 == 0:
            loss = calculate_logistic_loss(y, tx, w)
            print("iter: {}/{}, loss = {}\n".format(iter_,max_iters,loss))
            indices = random.sample(range(0, size_data), batch_size)
        grad = calculate_logistic_gradient(y[indices], tx[indices], w)
        w = w - gamma*grad
        
    loss = calculate_logistic_loss(y, tx, w)
    print("iter: {}/{}, loss = {}\n".format(iter_,max_iters,loss))
    return (w, loss)

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    '''Regularized Logistic regression with gradient descent'''

    w = initial_w
    for iter_ in range(max_iters):
        if iter_%1000 == 0:
            loss = calculate_logistic_loss(y, tx, w)
            print("iter: {}/{}, loss = {}\n".format(iter_,max_iters,loss))
        grad = calculate_logistic_gradient(y, tx, w) + 2*lambda_*w
        w = w - gamma*grad
        
    loss = calculate_logistic_loss(y, tx, w)    
    print("iter: {}/{}, loss = {}\n".format(iter_,max_iters,loss))
    return (w, loss)



