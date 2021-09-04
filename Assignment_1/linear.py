import numpy as np
import scipy as sp
import pandas as pd
import sys


def preprocess_dataset(dataset_file):
    data = pd.read_csv(dataset_file)
    data.drop("Unnamed: 0", inplace=True, axis=1)
    all_ones = [1 for _ in range(data.shape[0])]
    data.insert(loc=0, column="Ones", value=all_ones)

    X, y = data.iloc[:, : data.shape[1] - 1], data.iloc[:, -1:]
    X, y = X.to_numpy(), y.to_numpy()
    return X, y


def normal_equation(X, y):
    """
    This function implements the normal equation to calculate the weights.
    """
    weights = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    return weights


def ridge_regression(X, y, lambda_):
    """
    This function implements the ridge regression to calculate the weights.
    """
    X_t = np.transpose(X)
    lambda_identity = lambda_ * np.identity(X.shape[1])
    weights = np.linalg.inv(X_t @ X + lambda_identity) @ X_t @ y
    return weights


def train_test_split(X, y, k, i):
    n = X.shape[0]  # number of training examples
    X_train, X_test, y_train, y_test = (
        np.concatenate((X[: i * n // k], X[(i + 1) * n // k :])),
        X[i * n // k : (i + 1) * n // k],
        np.concatenate((y[: i * n // k], y[(i + 1) * n // k :])),
        y[i * n // k : (i + 1) * n // k],
    )
    return X_train, X_test, y_train, y_test


def k_folds_cross_validation(X, y, k, lambda_):
    """
    This function performs k-fold cross validation on the entire training set and returns optimal value of regularization parameter.
    X: Training examples
    y: labels
    k: number of folds
    lambda_: List of possible regularization parameters
    """
    optimal_lambda = lambda_[0]
    min_error = sys.maxsize
    optimal_weights = None
    test_set_size = X.shape[0] // k

    for reg_param in lambda_:
        error = 0
        for iteration in range(k):
            # Split the data into k folds
            X_train, X_test, y_train, y_test = train_test_split(X, y, k, iteration)
            # Fit the model
            weights = ridge_regression(X_train, y_train, reg_param)
            # Predict the labels
            y_pred = X_test.dot(weights)
            # Calculate the error
            error += np.sum(np.square(y_pred - y_test))
        error //= test_set_size
        if error < min_error:
            min_error = error
            optimal_lambda = reg_param
            optimal_weights = weights
    return optimal_lambda, optimal_weights


def predict(test_file, weights):
    """
    This function predicts the output for the test dataset.
    """
    data = pd.read_csv(test_file)
    data.drop("Unnamed: 0", inplace=True, axis=1)
    all_ones = [1 for _ in range(data.shape[0])]
    data.insert(loc=0, column="Ones", value=all_ones)

    X_test = data.to_numpy()
    y_pred = X_test.dot(weights)

    return y_pred


def write_array_to_file(arr, filename, write_mode="Array"):
    """
    Create a new file and write a numpy array/scalar to it.
    If write_mode is 'Array', then the array is saved as a line break-delimited file.
    Else, the scalar is saved as a single line.
    """
    if write_mode == "Array":
        with open(filename, "w") as f:
            for i in range(arr.shape[0]):
                f.write(str(arr[i][0]) + "\n")
    elif write_mode == "Scalar":
        with open(filename, "w") as f:
            f.write(str(arr) + "\n")


if __name__ == "__main__":
    mode = sys.argv[1]
    train_data_file, test_data_file = sys.argv[2], sys.argv[3]

    if mode == "a":
        output_file, weight_file = sys.argv[4], sys.argv[5]
        X_train, y_train = preprocess_dataset(train_data_file)
        weights = normal_equation(X_train, y_train)
        predictions = predict(test_data_file, weights)
        write_array_to_file(predictions, output_file)
        write_array_to_file(weights, weight_file)

    elif mode == "b":
        regularization_file, output_file, weight_file, best_param_file = (
            sys.argv[4],
            sys.argv[5],
            sys.argv[6],
            sys.argv[7],
        )
        lambda_ = [float(i) for i in open(regularization_file).readlines()]
        X_train, y_train = preprocess_dataset(train_data_file)
        optimal_lambda, optimal_weights = k_folds_cross_validation(
            X_train, y_train, 10, lambda_
        )
        predictions = predict(test_data_file, optimal_weights)
        write_array_to_file(predictions, output_file)
        write_array_to_file(optimal_weights, weight_file)
        write_array_to_file(optimal_lambda, best_param_file, write_mode="Scalar")

    elif mode == "c":
        output_file = sys.argv[4]
