# Mandatory functions
def least_squares_GD(y, tx, initial w, max_iters, gamma):
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

def least_squares_SGD(y, tx, initial w, max_iters, gamma):
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

def logistic_regression(y, tx, initial w, max_iters, gamma):
    raise NotImplementedError 


def reg_logistic_regression(y, tx, lambda_, initial w, max_iters, gamma):
    raise NotImplementedError



# Misc functions
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    error = np.subtract(y,(tx@w).flatten())
    grad = -(1/len(y))*np.transpose(tx)@error
    return grad


def build_poly_1D(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.c_[np.ones(x.shape),x]
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



