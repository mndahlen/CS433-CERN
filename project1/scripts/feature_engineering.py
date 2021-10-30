import numpy as np
from helpers import *

# LOAD DATA
DATA_TRAIN_PATH = '../data/train.csv'
DATA_EVAL_PATH = '../data/test.csv'

# GET TRAIN AND TEST DATA
y_train, x_train,  idx_train = load_csv_data(DATA_TRAIN_PATH, use_pandas=True, classes=[1, -1])
_, x_eval, idx_eval = load_csv_data(DATA_EVAL_PATH, use_pandas=True, classes=[1, -1])


def create_features(x_train, x_test):
    '''
        Creates features with three steps:
        1. Sets all outliers to mean of non-outlier data.
        2. Standardizes each feature by setting mean to 0 and normalizing standard deviation.
        3. Includes and combines raw features into first and second order polynomials
    '''
    x_train = handle_outliers(x_train)
    x_test = handle_outliers(x_test)

    x_train, mean_x, std_x = standardize(x_train)
    x_test, mean_x_test, std_x_test = standardize(x_test)

    poly_features = [[0],[0,1],[0,2],[0,7],[0,16],[1],[1,10],[0,0],[2,7],[2,2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[16],[16,16]]
    x_train = create_poly(x_train, poly_features)
    x_test = create_poly(x_test, poly_features)

    return x_train, x_test	


def create_poly(x, poly_features): 
    '''
        Creates features specified in poly_features.
        Each sub-list is one feature. The indexes in each subarray specifies what
        features (indexes) of x to combine multiplicatively.
    '''
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


def percentage_missing_data_per_feature(x, threshold):
    """
    One way to perform feature selection is to remove columns
    where there are too much missing values.
    """
    n, d = x.shape
    missing_values_percentages = []
    for col in range(d):
        missing_values = np.sum(x[:, col] == -999)
        missing_values_percentages.append(missing_values)
    missing_values_percentages = (1 / n) * np.array(missing_values_percentages)

    remove_idx = []
    for idx, percentage in enumerate(missing_values_percentages):
        if percentage > threshold:
            remove_idx.append(idx)
    print('Consider removing these indices from input_data with more than {}% missing values'.format(threshold * 100))
    print(remove_idx)
    return remove_idx


def low_variance_feature_in_raw_data(x_raw, threshold):
    """
    Another way to remove features to reduce dimensionality is to
    remove variables which have a low variance (of course before standardizing the data). It means that these
    values will struggle to explain the data if they don't vary too much.
    """
    n = x_raw.shape[0]
    features_variance = (1 / n) * np.sum(np.square(x_raw - np.mean(x_raw, axis=0)), axis=0)
    low_variance_idx = []
    for idx, variance in enumerate(features_variance):
        if variance < threshold:
            low_variance_idx.append(idx)
    print("Features variance in raw data :", features_variance)
    print('Consider removing these indices from raw input_data with a variance lower than {} threshold'.format(threshold))
    print(low_variance_idx)
    return low_variance_idx


def feature_selection_pairwise_correlation(x, threshold=0.8, remaining_feature=30):
    """
    One other way to perform feature selection is to analyze the correlation matrix.
    While doing a regression we don't want the features to be too much correlated to each other.
    Otherwise the information will be redundant and the computation time will increase with no precision gain.
    """
    corr_matrix = np.corrcoef(x, rowvar=False)
    d = corr_matrix.shape[0]
    redundant_features_idx = []
    features_cluster = dict()
    # Correlation matrix is symmetric :
    for i in range(d):
        for j in range(i + 1, d):
            if (np.abs(corr_matrix[i, j]) > threshold) and (i not in redundant_features_idx) and (j not in redundant_features_idx):
                redundant_features_idx.append(j)
                if i in features_cluster:
                    features_cluster[i].append(j)
                else:
                    features_cluster[i] = [j]
    print('The following dict values are the indices of features overlapping which have an absolute correlation coefficient above {} threshold.'.format(threshold))
    print('Consider removing these features, and keep their representative features whose indices are in dict keys.')
    print(features_cluster)
    return features_cluster


if __name__ == '__main__':
    percentage_missing_data_per_feature(x_train, threshold=0.4)
    low_variance_feature_in_raw_data(x_train, threshold=1.0)
    feature_selection_pairwise_correlation(x_train, threshold=0.8)
