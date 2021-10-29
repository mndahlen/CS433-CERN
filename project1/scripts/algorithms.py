import numpy as np
import time
from implementations import *


def ADAM(y, tx, lambda_, initial_w, max_iters, loss_function, gradient_function):
	"""Immplementation of ADAM."""
	w = initial_w
	alpha = 1.0
	beta1 = 0.9
	beta2 = 0.999
	eps = 1e-8
	m, v = np.zeros(np.shape(w)), np.zeros(np.shape(w))

	times = []
	losses = []
	Ws = [w]

	for iter in range(max_iters):

		tic = time.time()

		gradient = gradient_function(w)

		m_next = beta1 * m + (1 - beta1) * gradient
		v_next = beta2 * v + (1 - beta2) * np.square(gradient)

		m_hat = m_next / (1 - np.power(beta1, iter + 1))
		v_hat = v_next / (1 - np.power(beta2, iter + 1))

		H = np.sqrt(v_hat) + eps

		w_next = w - alpha * np.divide(m_hat, H)

		# Compute error and save data to be plotted later on.
		times.append(time.time() - tic)
		loss = loss_function(w)
		losses.append(loss)

		# Print the information.
		if (iter % 1000 == 0) or (iter == 0):
			print('Iter = {:4d},  Loss = {:0.9f}'.format(iter, loss))

		# Prepare the next iteration
		w = w_next
		Ws.append(w)
		m = m_next
		v = v_next

	loss = loss_function(w)
	losses.append(loss)
	best_iter = np.argmin(losses)

	return Ws[best_iter], losses[best_iter]


def LSAGDR(y, tx, lambda_, initial_w, max_iters, loss_function, gradient_function):
	"""Implementation of line search accelerated gradient descent with restart"""

	w = initial_w
	maxit = max_iters
	L = lipshitz_constant(tx, lambda_)
	z = w
	t = 1.0

	losses = []
	Ws =[w]
	times = []

	for iter in range(maxit):
		# Start the clock.
		tic = time.time()

		d = - gradient_function(z)
		L0 = L / 2
		i = 0

		while True:
			if loss_function(z + (1 / ((2**i) * L0)) * d) <= loss_function(z) - (1 / ((2**(i + 1)) * L0)) * np.linalg.norm(d)**2:
				L_next = (2**i) * L0
				alpha = 1 / L_next
				break
			i = i + 1

		w_next = z - alpha * gradient_function(z)

		if loss_function(w) < loss_function(w_next):
			z, t = w, 1.0
			d = - gradient_function(z)
			i = 0
			while True:
				if loss_function(z + (1 / ((2**i) * L0)) * d) <= loss_function(z) - (1 / ((2**(i + 1)) * L0)) * np.linalg.norm(d)**2:
					L_next = (2**i) * L0
					alpha = 1 / L_next
					break
				i = i + 1
			w_next = z - alpha * gradient_function(z)

		t_next = (1 / 2) * (1 + np.sqrt(1 + 4 * (L_next / L) * t**2))
		z_next = w_next + ((t - 1) / t_next) * (w_next - w)

		# Compute error and save data to be plotted later on.
		times.append(time.time() - tic)
		losses.append(loss_function(w))

		# Print the information.
		if (iter % 5 == 0) or (iter == 0):
			print('Iter = {:4d},  Loss = {:0.9f}'.format(iter, losses[-1]))

		# Prepare the next iteration
		w = w_next
		Ws.append(w)
		t = t_next
		z = z_next
		L = L_next

	losses.append(loss_function(w))
	best_iter = np.argmin(losses)

	return Ws[best_iter], losses[best_iter]


def lipshitz_constant(tx, mu=0.0):
	return (1 / 2) * np.linalg.norm(tx, 'fro') ** 2 + mu


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
