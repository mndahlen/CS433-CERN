import numpy as np

# Mandatory functions

# Should be okay according to requirements. Not tested
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    w = initial_w
    for n_iter in range(max_iters):
        #loss = compute_loss(y,tx,w)
        G = compute_gradient(y,tx,w)
        # Update Weights with Gradient Descent
        w = w - gamma*G
    loss = compute_mse(y, tx, w)
    return (w, loss)

# Should be okay according to requirements. Not tested
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    w = initial_w
    for n_iter in range(max_iters):
        # Choose one random index for descent. Default according to requirements
        idx = np.random.randint(len(y))
        #Stoch_G = compute_gradient(np.array(y[idx]), np.array(tx[idx]), w)
        e = y[idx] - tx[idx].dot(w)
        Stoch_G = -tx[idx]*e
        # Update Weights with Stochastic Gradient Descent
        w = w - gamma*Stoch_G
    loss = compute_mse(y, tx, w)
    return (w, loss)

# Should be okay according to requirements. Not tested
def least_squares(y, tx):
    """calculate the least squares solution."""
    #w_star = np.linalg.inv(inner_product)@np.transpose(tx)@y
    w = np.linalg.solve(np.transpose(tx)@tx, np.transpose(tx)@y)
    loss = compute_mse(y, tx, w)
    return (w, loss)

# Should be okay according to requirements. Not tested
def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(y)
    lambda_prime = 2*N*lambda_
    w = np.linalg.solve(np.transpose(tx)@tx + lambda_prime*np.eye(tx.shape[1]),np.transpose(tx)@y)
    loss = compute_rmse(y,tx,w)
    return (w, loss)

# Should be okay according to requirements. Not tested
# Logistic regression with gradient descent
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for iter in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w)
        w = w - gamma*grad
    loss = calculate_logistic_loss(y, tx, w)
    return (w, loss)

# Should be okay according to requirements. Not tested
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # start the logistic regression
    for iter in range(max_iters):
        grad = calculate_logistic_gradient(y, tx, w) + 2*lambda_*w
        w = w - gamma*grad
    loss = calculate_logistic_loss(y, tx, w)
    return (w, loss)

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

def calculate_logistic_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    loss = 0
    for i, x_i in enumerate(tx):
        sig = sigmoid(x_i.dot(w))
        loss -= y[i]*np.log(sig) + (1-y[i])*np.log(1-sig)
    return loss[0]

def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    grad = 0
    for i, x_i in enumerate(tx):
        sig = sigmoid(x_i.dot(w))
        grad -= (y[i]-sig)*x_i
    return grad



