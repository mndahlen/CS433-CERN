from helpers import load_csv_data
import numpy as np

def create_features(x_train,x_test):
    # i = 0
    # for feature in range(0,x_train.shape[1]):
    #     col_train = x_train[:,feature]
    #     col_test = x_test[:,feature]
    #     print(col_train[col_train==-999].shape,col_test[col_test==-999].shape,i)
    #     i += 1

    x_train = handle_outliers(x_train)
    x_test = handle_outliers(x_test)

    x_train, mean_x, std_x = standardize(x_train)
    x_test, mean_x_test, std_x_test = standardize(x_test)

    poly_features = [[0],[1],[2,2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13]]
    x_train = create_poly(x_train,poly_features)
    x_test = create_poly(x_test,poly_features)

    return x_train, x_test	

def create_poly(x_train,poly_features):
    rows = x_train.shape[0]
    features = []
    for feature in poly_features:
        cur = np.ones([rows,1])
        for col in feature:
            cur = np.multiply(cur,(x_train[:,col]).reshape(rows,1))
        features.append(cur.flatten())

    return np.transpose(np.asarray(features))

def handle_outliers(data, remove_outliers = False):
	for i in range(0,data.shape[1]):
		col = data[:,i]
		mean = np.mean(col[col != -999])
		col[col == -999] = mean
		data[:,i] = col

	return data

def standardize_data(data):
	for i in range(data.shape[1]):
		data[1:, i] = (data[1:, i] - np.mean(data[1:, i])) / np.std(data[1:, i])
	return None

def augment(x, powers):
    augmented = []
    for i in range(0,len(powers)):
        P = powers[i]
        for p in range(1,P+1):
            augmented.append(np.power(x[:,i],p))
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

if __name__ == '__main__':
	yb, input_data, ids = load_csv_data('../data/train.csv')
	deal_with_outliers(input_data)
	standardize_data(input_data)

