import numpy as np
from algorithms import *
from helpers import *


def train_model(y_train, x_train, hyperparameters, model='logistic_regression', algorithm='GD'):
        model_to_train = build_model(model, algorithm, hyperparameters)
        w, loss = model_to_train(y_train, x_train)
        return w, loss

def build_model(model, algo, hyperparameters):
	assert model in ['logistic_regression', 'l2_reg_logistic_regression'], 'The model is not available.'
	assert algo in ['GD', 'SGD', 'ADAM', 'LSAGDR'], 'The optimization algorithm is not available'

	gamma = hyperparameters["gamma"]
	initial_w = hyperparameters["initial_w"]
	max_iters = hyperparameters["max_iters"]
	conv_limit = hyperparameters["conv_limit"]
	lambda_ = hyperparameters["lambda"]

	if algo == 'GD':
		if model == 'logistic_regression':
			return lambda y, tx: logistic_regression(y, tx, initial_w, max_iters, gamma, conv_limit)
		if model == 'l2_reg_logistic_regression':
			return lambda y, tx: reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, conv_limit)

	if algo == 'SGD':
		if model == 'logistic_regression':
			return lambda y, tx: logistic_regression_SGD(y, tx, initial_w, max_iters, gamma, conv_limit)
		if model == 'l2_reg_logistic_regression':
			return lambda y, tx: reg_stochastic_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, conv_limit)

	if algo == 'ADAM':
		return lambda y, tx: ADAM(y, tx, lambda_, initial_w, max_iters, lambda w: compute_neg_log_likelihood_l2(y, tx, w, lambda_), lambda w: compute_gradient_log_reg_l2(y, tx, w, lambda_))

	if algo == 'LSAGDR':
		return lambda y, tx: LSAGDR(y, tx, lambda_, initial_w, max_iters, lambda w: compute_neg_log_likelihood_l2(y, tx, w, lambda_), lambda w: compute_gradient_log_reg_l2(y, tx, w, lambda_))

	return None