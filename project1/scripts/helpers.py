# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import pandas as pd
import time

def load_csv_data(data_path, sub_sample=False, use_pandas = False, classes = [1,-1]):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    t0 = time.time()
    if use_pandas:
        print("Loading data with Pandas")
        data = pd.read_csv(data_path)
        data.loc[data['Prediction'] == 's','Prediction'] = classes[0]
        data.loc[data['Prediction'] == 'b','Prediction'] = classes[1]
        data.loc[data['Prediction'] == '?','Prediction'] = -10
        data['Prediction'] = pd.to_numeric(data['Prediction'])
        yb = data['Prediction'].to_frame().to_numpy().flatten()
        ids = data['Id'].to_frame().to_numpy()
        data.drop(['Id','Prediction'], 1, inplace=True)
        input_data = data.to_numpy()

    else:
        print("Loading data with slow given function")
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
    print("Finally completed loading data!\nThat was {} seconds!".format(int(time.time() - t0)))

    return yb, input_data, ids


def predict_labels(tx, w, classes = [1,-1]):
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
    augmented = []
    for i in range(0,len(powers)):
        P = powers[i]
        for p in range(1,P+1):
            augmented.append(np.power(x[:,i],p))
    return np.transpose(np.asarray(augmented))

def create_features(x_train,x_test):
    remove_vars = [2, 4, 7, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    augment_vars = [3, 2, 1, 1, 2, 3, 3, 2]

    x_train, mean_x, std_x = standardize(x_train)
    x_test, mean_x_test, std_x_test = standardize(x_test)
    x_train = remove_variables(x_train, remove_vars)
    x_test = remove_variables(x_test, remove_vars)
    x_train = augment(x_train, augment_vars)
    x_test = augment(x_test, augment_vars)
    
    return x_train, x_test
        
# Misc functions
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx.dot(w)
    return -1/(y.shape[0])*(np.transpose(tx)).dot(e)

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
def compute_rmse(y,tx,w):
    error = np.subtract(y,(tx@w))
    N = len(y)
    MSE = np.transpose(error)@error/(2*N)
    loss = np.sqrt(2 * MSE)
    return loss

def compute_mse(y, tx, w):
    e = y - np.dot(tx, w)
    return 1/(2*y.shape[0])*np.transpose(e).dot(e)

def compute_mae(y,tx,w):
    error = np.subtract(y,(tx@w).flatten())
    N = len(y)
    loss = np.sum(np.absolute(error))/N
    return loss

# For logistic regression
def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1/(1 + np.exp(-t))

def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    sig = sigmoid(tx@w)
    grad_coefficient = sig - y
    grad = np.transpose(tx)@grad_coefficient
    
    return grad

def calculate_logistic_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    sig = sigmoid(tx@w)
    N = (y.shape)[0]
    loss = np.transpose(y)@np.log(sig) + np.transpose(np.ones(y.shape) - y)@np.log(np.ones(sig.shape) - sig)
    
    return -loss/N

