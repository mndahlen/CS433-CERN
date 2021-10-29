# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import pandas as pd
import time


def load_csv_data(data_path, sub_sample=False, use_pandas=False, classes=[1, -1]):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    t0 = time.time()
    if use_pandas:
        print("Loading data with Pandas")
        data = pd.read_csv(data_path)
        data.loc[data['Prediction'] == 's', 'Prediction'] = classes[0]
        data.loc[data['Prediction'] == 'b', 'Prediction'] = classes[1]
        data.loc[data['Prediction'] == '?', 'Prediction'] = -10
        data['Prediction'] = pd.to_numeric(data['Prediction'])
        yb = data['Prediction'].to_frame().to_numpy().flatten()
        ids = data['Id'].to_frame().to_numpy()
        data.drop(labels=['Id', 'Prediction'], axis=1, inplace=True)
        input_data = data.to_numpy()

    else:
        print("Loading data with slow given function")
        y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
        x = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=float)
        ids = x[:, 0].astype(np.int)
        input_data = x[:, 2:]

        # convert class labels from strings to binary (-1,1)
        yb = classes[0] * np.ones(len(y))
        yb[np.where(y == 'b')] = classes[1]
        
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]
    print("Finally completed loading data!\nThat was {} seconds!".format(int(time.time() - t0)))

    return yb, input_data, ids


def predict_labels(tx, w, classes=[1, -1]):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = tx@w
    y_pred[np.where(y_pred > 0)] = classes[0]
    y_pred[np.where(y_pred <= 0)] = classes[1]
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


# Remove varibles at indexes indicated in remove_idxs
def remove_variables(x, remove_idxs):
    return np.delete(x, remove_idxs, axis=1)


def build_poly_1D(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.c_[np.ones(x.shape[0]), x]
    for d in range(2, degree + 1):
        poly = np.c_[poly, np.power(x, d)]
    return poly


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # Set seed and shuffle data
    np.random.seed(seed)
    x_w_output = np.c_[x, y]
    np.random.shuffle(x_w_output)
    x_shuffled = x_w_output[:, :-1]
    y_shuffled = x_w_output[:, -1]
    # Prepare split index
    split_index = int(np.round(x_shuffled.shape[0] * ratio))

    # Split
    x_train = x_shuffled[:split_index]
    y_train = y_shuffled[:split_index]
    x_test = x_shuffled[split_index:]
    y_test = y_shuffled[split_index:]

    return x_train, y_train, x_test, y_test


# Score functions
def F1_score(y_pred, y_train):
    tp = np.sum(y_pred[np.where(y_train == y_pred)] > 0)
    fp = np.sum(y_pred > 0) - tp
    fn = np.sum(y_pred[np.where(y_train != y_pred)] <= 0)
    return tp / (tp + (1 / 2) * (fp + fn))


def cat_accuracy(y_pred, y_train):
    return np.sum(y_pred == y_train) / len(y_train)
