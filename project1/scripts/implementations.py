import numpy as np

# Mandatory functions
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y,tx,w)
        G = compute_gradient(y,tx,w)
        
        # Update Weights with Gradient Descent
        w = w - gamma*G

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return (w, loss)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            loss = compute_loss(y,tx,w)
            Stoch_G = compute_gradient(minibatch_y,minibatch_tx,w)
            break
            
        # Update Weights with Stochastic Gradient Descent
        w = w - gamma*Stoch_G
        losses.append(loss)
        ws.append(w)
    return (w, loss)

def least_squares(y, tx):
    """calculate the least squares solution."""
    #w_star = np.linalg.inv(inner_product)@np.transpose(tx)@y
    w = np.linalg.solve(np.transpose(tx)@tx, np.transpose(tx)@y)
    loss = compute_mse(y, tx, w)
    return (w, loss)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    N = len(y)
    lambda_prime = 2*N*lambda_
    w = np.linalg.solve(np.transpose(tx)@tx + lambda_prime*np.eye(tx.shape[1]),np.transpose(tx)@y)
    loss = compute_rmse(y,tx,w)
    return (w, loss)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = logistic_gradient_descent(y, tx, w, gamma)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return (w, loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    #max_iter = 10000
    #gamma = 0.01
    #lambda_ = 0.1
    threshold = 1e-8
    losses = []

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = penalized_logistic_gradient_descent(y, tx, w, gamma, lambda_)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return (w, loss)



# Misc functions
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    error = np.subtract(y,(tx@w).flatten())
    grad = -(1/len(y))*np.transpose(tx)@error
    return grad


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
    x_shuffled = x_w_output[:,0]
    y_shuffled = x_w_output[:,1]

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
    error = np.subtract(y,(tx@w).flatten())
    N = len(y)
    loss = error.transpose()@error/N
    return loss


def compute_mae(y,tx,w):
    error = np.subtract(y,(tx@w).flatten())
    N = len(y)
    loss = np.sum(np.absolute(error))/N
    return loss

# For logistic regression
def sigmoid(t):
    """apply the sigmoid function on t."""
    sig = np.power((np.ones(t.shape) + np.exp(-t)),-1)
    return sig

def calculate_logistic_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    sig = sigmoid(tx@w)
    loss = np.transpose(y)@np.log(sig) + np.transpose(np.ones(y.shape) - y)@np.log(np.ones(sig.shape) - sig)
    
    return -loss

def calculate_logistic_gradient(y, tx, w):
    """compute the gradient of loss."""
    common_sigmoid = sigmoid(tx@w)
    grad_coefficient = common_sigmoid - y
    grad = np.transpose(tx)@grad_coefficient # Wrong way?
    
    return grad

def logistic_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w)
    
    grad = calculate_gradient(y,tx,w)

    w = w - gamma*grad
    
    return loss, w

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    grad = calculate_logistic_gradient(y, tx, w) + 2*lambda_*w
    loss = calculate_logistic_loss(y, tx, w) + lambda_*(np.transpose(w)@w)
    return loss, grad

def penalized_logistic_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad = penalized_logistic_regression(y,tx,w,lambda_)
    w = w - gamma*grad

    return loss, w



