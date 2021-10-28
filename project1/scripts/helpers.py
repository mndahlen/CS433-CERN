# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=float)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = 0
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
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
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

# Standard, and destandardazing methods from lab5. To use with visualization
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x

# Remove varibles at indexes indicated in remove_idxs
def remove_variables(x, remove_idxs):
    return np.delete(x, remove_idxs, axis=1)

# Insert zeros afterwards to be able to run tests
# Send with x as reference for size
# Yeah, really b function. I was tired and this is only run once
def insert_zeros(w, size, insert_idxs):
    w_idx = 0
    new_w = []
    for idx in range(size):
        if idx in insert_idxs:
            new_w.append(0)
        else:
            new_w.append(w[w_idx])
    return new_w

    # Do this with loop for simplicity
    
    return np.insert(x, insert_idxs, 0, axis=1)

def augment(x, powers):
    ret_x = []
    for i, xi in enumerate(x):
        new_xi = []
        for j, pw in enumerate(powers):
            new_xi = new_xi + [xi[j]**exp for exp in range(1, int(pw) + 1)]
        ret_x.append(new_xi)
    return np.array(ret_x)
    

    
