import numpy as np


def create_features(x_train, x_test):

    x_train = handle_outliers(x_train)
    x_test = handle_outliers(x_test)

    x_train, mean_x, std_x = standardize(x_train)
    x_test, mean_x_test, std_x_test = standardize(x_test)

    # Gave 78% [[0],[1],[2,2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13]]
    # Gave 79.5% [[0],[0,1],[0,2],[0,7],[0,16],[1],[1,10],[0,0],[2,7],[2,2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13]]
    # gave 79.8% [[0],[0,1],[0,2],[0,7],[0,16],[1],[1,10],[0,0],[2,7],[2,2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[16],[16,16],[17],[18],[20],[24],[25]]
    # poly_features = [[0],[0,1],[0,2],[0,7],[0,16],[1],[1,10],[0,0],[2,7],[2,2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[16],[16,16],[17],[18],[20],[24],[25]]
    poly_features = [[0],[0,1],[0,2],[0,7],[0,16],[1],[1,10],[0,0],[2,7],[2,2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[16],[16,16]]
    x_train = create_poly(x_train, poly_features)
    x_test = create_poly(x_test, poly_features)

    return x_train, x_test	


def create_poly(x, poly_features):
    rows = x.shape[0]

    features = []
    for feature in poly_features:
        cur = np.ones([rows, 1])
        for col in feature:
            cur = np.multiply(cur, (x[:, col]).reshape(rows, 1))
        features.append(cur.flatten())

    return np.transpose(np.asarray(features))


def handle_outliers(data):
    '''
       Takes care of outliers in AIcrowd higgs boson data
       by setting outliers marked with -999 to mean of
       non-outlier data.
    '''
    for i in range(0, data.shape[1]):
        col = data[:, i]
        mean = np.mean(col[col != -999])
        col[col == -999] = mean
        data[:, i] = col

    return data


def augment(x, powers):
    augmented = []
    for i in range(0, len(powers)):
        P = powers[i]
        for p in range(1, P+1):
            augmented.append(np.power(x[:, i], p))
    return np.transpose(np.asarray(augmented))


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
