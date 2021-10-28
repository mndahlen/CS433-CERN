import numpy as np
import time
from helpers import load_csv_data, predict_labels
from data_engineering import deal_with_outliers, standardize_data
from implementations import compute_gradient_log_reg_l2, compute_neg_log_likelihood_l2, logistic_regression

data_path = '../data/train.csv'


def ADAM(y, tx, lambda_, initial_w, max_iters):
	"""
	Immplementation of ADAM.
	:param fx:
	:param gradf:
	:param parameter:
	:return:
	"""
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

		gradient = compute_gradient_log_reg_l2(y, tx, w, lambda_)

		m_next = beta1 * m + (1 - beta1) * gradient
		v_next = beta2 * v + (1 - beta2) * np.square(gradient)

		m_hat = m_next / (1 - np.power(beta1, iter + 1))
		v_hat = v_next / (1 - np.power(beta2, iter + 1))

		H = np.sqrt(v_hat) + eps

		w_next = w - alpha * np.divide(m_hat, H)

		# Compute error and save data to be plotted later on.
		times.append(time.time() - tic)
		loss = compute_neg_log_likelihood_l2(y, tx, w, lambda_)
		losses.append(loss)

		# Print the information.
		if (iter % 100 == 0) or (iter == 0):
			print('Iter = {:4d},  Loss = {:0.9f}'.format(iter, loss))

		# Prepare the next iteration
		w = w_next
		Ws.append(w)
		m = m_next
		v = v_next

	loss = compute_neg_log_likelihood_l2(y, tx, w, lambda_)
	losses.append(loss)

	return Ws, losses


def LSAGDR(y, tx, lambda_, initial_w, max_iters):
	"""
	Function:  [x, info] = LSAGDR (fx, gradf, parameter)
	Purpose:   Implementation of AGD with line search and adaptive restart.
	Parameter: x0         - Initial estimate.
		   maxit      - Maximum number of iterations.
		   Lips       - Lipschitz constant for gradient.
		   strcnvx    - Strong convexity parameter of f(x).
	:param fx:
	:param gradf:
	:param parameter:
	:return:
	"""

	w = initial_w
	maxit = max_iters
	L = lipshitz_constant(tx, lambda_)
	z = w
	t = 1.0
	loss = lambda w: compute_neg_log_likelihood_l2(y, tx, w, lambda_)
	gradient = lambda w: compute_gradient_log_reg_l2(y, tx, w, lambda_)

	losses = []
	Ws =[w]
	times = []

	for iter in range(maxit):
		# Start the clock.
		tic = time.time()

		d = - gradient(z)
		L0 = L / 2
		i = 0

		while True:
			if loss(z + (1 / ((2**i) * L0)) * d) <= loss(z) - (1 / ((2**(i + 1)) * L0)) * np.linalg.norm(d)**2:
				L_next = (2**i) * L0
				alpha = 1 / L_next
				break
			i = i + 1

		w_next = z - alpha * gradient(z)

		if loss(w) < loss(w_next):
			z, t = w, 1.0
			d = - gradient(z)
			i = 0
			while True:
				if loss(z + (1 / ((2**i) * L0)) * d) <= loss(z) - (1 / ((2**(i + 1)) * L0)) * np.linalg.norm(d)**2:
					L_next = (2**i) * L0
					alpha = 1 / L_next
					break
				i = i + 1
			w_next = z - alpha * gradient(z)

		t_next = (1 / 2) * (1 + np.sqrt(1 + 4 * (L_next / L) * t**2))
		z_next = w_next + ((t - 1) / t_next) * (w_next - w)

		# Compute error and save data to be plotted later on.
		times.append(time.time() - tic)
		losses.append(loss(w))

		# Print the information.
		if (iter % 5 == 0) or (iter == 0):
			print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, losses[-1]))

		# Prepare the next iteration
		w = w_next
		Ws.append(w)
		t = t_next
		z = z_next
		L = L_next

	losses.append(loss(w))

	return Ws, losses


def lipshitz_constant(tx, mu=0.0):
	return (1 / 2) * np.linalg.norm(tx, 'fro') ** 2 + mu


if __name__ == '__main__':
	print('Loading data')
	y, X, ids = load_csv_data('../data/train.csv', sub_sample=False)
	deal_with_outliers(X)
	standardize_data(X)
	print('Data loaded')

	max_iters = 1000
	threshold = 1e-8
	gamma = 0.01
	lambda_ = 0.1
	losses = []
	y = y.reshape((-1, 1))  # careful
	tx = np.c_[np.ones((y.shape[0], 1)), X]
	initial_w = np.zeros((tx.shape[1], 1))

	print('Starting training')
	# Ws, losses = ADAM(y, tx, lambda_, initial_w, max_iters)
	Ws, losses = LSAGDR(y, tx, lambda_, initial_w, max_iters)
	# w, loss = logistic_regression(y, tx, initial_w, max_iters, gamma)

	print('Ws =', len(Ws))
	print('losses =', len(losses))
	w = Ws[np.argmin(losses)]
	print('Training done')

	y_pred = predict_labels(tx, w)
	acc = np.sum(y_pred == y) / len(y)
	print('Accuracy = ', acc)
