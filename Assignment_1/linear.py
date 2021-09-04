import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sys


def preprocess_dataset(dataset_file):
    data = pd.read_csv(dataset_file)
    data.drop("Unnamed: 0", inplace=True, axis=1)
    all_ones = [1 for _ in range(data.shape[0])]
    data.insert(loc=0, column="Ones", value=all_ones)

    X, y = data.iloc[:, : data.shape[1] - 1], data.iloc[:, -1:]
    X, y = X.to_numpy(), y.to_numpy()
    return X, y


def normalEquation(X, y):
    """
    This function implements the normal equation to calculate the weights.
    """
    weights = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y
    return weights


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


def write_array_to_file(arr, filename):
    """
    Create a new file and write a numpy array to it
    """
    with open(filename, "w") as f:
        for i in range(arr.shape[0]):
            f.write(str(arr[i][0]) + "\n")


# train_data_file = "./data/train.csv"

if __name__ == "__main__":
    mode = sys.argv[1]
    train_data_file, test_data_file = sys.argv[2], sys.argv[3]

    if mode == "a":
        output_file, weight_file = sys.argv[4], sys.argv[5]
        X_train, y_train = preprocess_dataset(train_data_file)
        weights = normalEquation(X_train, y_train)
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
        # do something
    elif mode == "c":
        output_file = sys.argv[4]
