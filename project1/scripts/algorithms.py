import numpy as np
from helpers import load_csv_data
from data_engineering import deal_with_outliers, standardize_data
from implementations import reg_stochastic_logistic_regression

data_path = '../data/train.csv'

if True:
	print('Loading data')
	y, X, ids = load_csv_data('../data/train.csv', sub_sample=True)
	deal_with_outliers(X)
	standardize_data(X)
	print('Data loaded')

	max_iters = 300
	threshold = 1e-8
	gamma = 0.01
	lambda_ = 0.1
	losses = []
	y = y.reshape((-1, 1))  # careful
	tx = np.c_[np.ones((y.shape[0], 1)), X]
	initial_w = np.zeros((tx.shape[1], 1))

	print('Starting training')
	w, loss = reg_stochastic_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
	print(w.shape)
	print(loss)
	print('Training done')
