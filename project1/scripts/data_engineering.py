from helpers import load_csv_data
import numpy as np


def deal_with_outliers(data):
	for i in range(data.shape[1]):
		col = data[1:, i]
		data[1:, i] = np.multiply((col != -999.00), col) + np.mean(col[np.where((col != -999.00))[0]]) * (col == -999.00)
	return None


def standardize_data(data):
	for i in range(data.shape[1]):
		data[1:, i] = (data[1:, i] - np.mean(data[1:, i])) / np.std(data[1:, i])
	return None


if __name__ == '__main__':
	yb, input_data, ids = load_csv_data('../data/train.csv')
	deal_with_outliers(input_data)
	standardize_data(input_data)

