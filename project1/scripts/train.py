import numpy as np
from algorithms import *
from helpers import *


def train_model(y_train, x_train, hyperparameters, model='logistic_regression', algorithm='GD'):

        model_to_train = build_model(model, algorithm, hyperparameters)
        w, loss = model_to_train(y_train, x_train)
        return w, loss
