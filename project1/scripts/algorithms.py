import numpy as np
import time
from helpers import load_csv_data, predict_labels
from data_engineering import deal_with_outliers, standardize_data
from implementations import compute_gradient_log_reg_l2, compute_neg_log_likelihood_l2

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
		m = m_next
		v = v_next

	return w, loss


if __name__ == '__main__':
	print('Loading data')
	y, X, ids = load_csv_data('../data/train.csv', sub_sample=False)
	deal_with_outliers(X)
	standardize_data(X)
	print('Data loaded')

	max_iters = 5000
	threshold = 1e-8
	gamma = 0.01
	lambda_ = 0.1
	losses = []
	y = y.reshape((-1, 1))  # careful
	tx = np.c_[np.ones((y.shape[0], 1)), X]
	initial_w = np.zeros((tx.shape[1], 1))

	print('Starting training')
	w, loss = ADAM(y, tx, lambda_, initial_w, max_iters)
	print('Loss =', loss)
	print('Training done')

	y_pred = predict_labels(tx, w)
	acc = np.sum(y_pred == y) / len(y)
	print('Accuracy = ', acc)
