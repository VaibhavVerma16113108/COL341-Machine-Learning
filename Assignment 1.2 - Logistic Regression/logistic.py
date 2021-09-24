import numpy as np
from scipy.special import softmax
import pandas as pd
import sys, math
import matplotlib.pyplot as plt


def read_and_encode(train_path, test_path):
    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)

    y_train = np.array(pd.get_dummies(train["Length of Stay"]))

    train = train.drop(columns=["Length of Stay"])
    # Ensuring consistency of One-Hot Encoding

    data = pd.concat([train, test], ignore_index=True)
    cols = train.columns
    cols = cols[:-1]
    data = pd.get_dummies(data, columns=cols, drop_first=True)
    data = data.to_numpy()
    X_train = data[: train.shape[0], :]
    X_test = data[train.shape[0] :, :]

    # Add bias columns
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    return X_train, y_train, X_test


def loss(X, y, W):
    """
    Compute the log-likelihood expression given the data X, y, and weights w.
    """
    inner_term = np.log(softmax(np.dot(X, W), axis=1))
    return -np.sum(np.sum(np.multiply(y, inner_term))) / (2 * X.shape[0])


def predict(X, W):
    """
    Predict the class of each sample in X using the weights w.
    """
    return softmax(np.dot(X, W), axis=1)


def update_weights(X, y, W, eta):
    W = W - eta * gradient(X, y, W)
    return W


def gradient(X, y, W):
    """
    Compute the gradient of the loss function with respect to the weights w.
    """
    return np.dot(X.T, (predict(X, W)) - y) / X.shape[0]


def alpha_beta_backtracking(X_train, y_train, W, eta, alpha, beta):
    """
    Perform backtracking line search to find the best learning rate.
    """
    pass


def update_learning_rate(X, y, W, update_type, eta, k, eta0, alpha, beta):
    if update_type == 1:
        return eta
    elif update_type == 2:
        return eta0 / math.sqrt(k)
    elif update_type == 3:
        return alpha_beta_backtracking(X, y, W, eta, alpha, beta)


def create_batch(X, y, batch_size, i):
    n = X.shape[0]  # number of training examples
    X_batch, y_batch = (
        # np.concatenate((X[: i * batch_size], X[(i + 1) * batch_size :])),
        X[(i - 1) * batch_size : i * batch_size],
        # np.concatenate((y[: i * batch_size], y[(i + 1) * batch_size :])),
        y[(i - 1) * batch_size : i * batch_size],
    )
    return X_batch, y_batch


def gradient_descent(
    X,
    y,
    eta,
    max_iters,
    mode="batch",
    update_type=None,
    eta0=None,
    alpha=None,
    beta=None,
):
    """
    Perform gradient descent to learn the weights of the model.
    """
    num_classes = y.shape[1]
    # Initialize the weights
    W = np.zeros((X.shape[1], num_classes))
    losses = []
    # Perform gradient descent
    if mode == "batch":
        for iteration in range(1, max_iters + 1):
            print("Iteration: ", iteration)
            W = update_weights(X, y, W, eta)
            eta = update_learning_rate(
                X, y, W, update_type, eta, iteration + 1, eta0, alpha, beta
            )
            print(eta)
    elif mode == "minibatch":
        for iteration in range(1, max_iters + 1):
            print("Iteration: ", iteration)
            for i in range(1, X.shape[0] // batch_size + 1):
                X_batch, y_batch = create_batch(X, y, batch_size, i)
                W = update_weights(X_batch, y_batch, W, eta)
            eta = update_learning_rate(
                X, y, W, update_type, eta, iteration + 1, eta0, alpha, beta
            )
            print(eta)
    print(loss(X, y, W))
    return W, losses


def write_to_file(arr, filename, write_mode="Array"):
    """
    Create a new file and write a numpy array/scalar to it.
    If write_mode is 'Array', then the array is saved as a line break-delimited file.
    Else, the scalar is saved as a single line.
    """
    if write_mode == "Array":
        with open(filename, "w") as f:
            for i in range(arr.shape[0]):
                f.write("{:.18e}".format(float(str(arr[i]))) + "\n")
    elif write_mode == "Scalar":
        with open(filename, "w") as f:
            f.write(str(arr) + "\n")


if __name__ == "__main__":
    mode = sys.argv[1]
    print(mode)
    train_data_file, test_data_file = sys.argv[2], sys.argv[3]

    if mode == "a":
        X_train, y_train, X_test = read_and_encode(train_data_file, test_data_file)
        print(X_train.shape, y_train.shape, X_test.shape)
        param_file, output_file, weight_file = sys.argv[4:]
        lines = open(param_file).readlines()
        type_of_strategy = int(lines[0].strip())
        if type_of_strategy == 1:
            eta = float(lines[1].strip())
            max_iters = int(lines[2].strip())
            weights, losses = gradient_descent(
                X_train, y_train, eta, max_iters, update_type=1
            )
            predictions = (
                np.argmax(softmax(np.dot(X_test, weights), axis=1), axis=1) + 1
            )
            print(predictions)
            write_to_file(predictions, output_file)
            write_to_file(weights.flatten(), weight_file)
        elif type_of_strategy == 2:
            eta0 = float(lines[1].strip())
            max_iters = int(lines[2].strip())
            weights, losses = gradient_descent(
                X_train, y_train, eta0, max_iters, update_type=2, eta0=eta0
            )
            predictions = (
                np.argmax(softmax(np.dot(X_test, weights), axis=1), axis=1) + 1
            )
            print(predictions)
            write_to_file(predictions, output_file)
            write_to_file(weights.flatten(), weight_file)
        else:
            eta, alpha, beta = [float(x) for x in lines[1].strip().split(",")]
            max_iters = int(lines[2].strip())
            weights, losses = gradient_descent(
                X_train, y_train, eta, max_iters, update_type=3, alpha=alpha, beta=beta
            )
            predictions = (
                np.argmax(softmax(np.dot(X_test, weights), axis=1), axis=1) + 1
            )
            print(predictions)
            write_to_file(predictions, output_file)
            write_to_file(weights.flatten(), weight_file)

    elif mode == "b":
        X_train, y_train, X_test = read_and_encode(train_data_file, test_data_file)
        print(X_train.shape, y_train.shape, X_test.shape)
        param_file, output_file, weight_file = sys.argv[4:]
        lines = open(param_file).readlines()
        type_of_strategy = int(lines[0].strip())
        max_iters = int(lines[2].strip())
        batch_size = int(lines[3].strip())
        if type_of_strategy == 1:
            eta = float(lines[1].strip())
            max_iters = int(lines[2].strip())
            weights, losses = gradient_descent(
                X_train, y_train, eta, max_iters, mode="minibatch", update_type=1
            )
            predictions = (
                np.argmax(softmax(np.dot(X_test, weights), axis=1), axis=1) + 1
            )
            print(predictions)
            write_to_file(predictions, output_file)
            write_to_file(weights.flatten(), weight_file)
        elif type_of_strategy == 2:
            eta0 = float(lines[1].strip())
            max_iters = int(lines[2].strip())
            weights, losses = gradient_descent(
                X_train,
                y_train,
                eta0,
                max_iters,
                mode="minibatch",
                update_type=2,
                eta0=eta0,
            )
            predictions = (
                np.argmax(softmax(np.dot(X_test, weights), axis=1), axis=1) + 1
            )
            print(predictions)
            write_to_file(predictions, output_file)
            write_to_file(weights.flatten(), weight_file)
        else:
            eta, alpha, beta = [float(x) for x in lines[1].strip().split(",")]
            max_iters = int(lines[2].strip())
            weights, losses = gradient_descent(
                X_train,
                y_train,
                eta,
                max_iters,
                mode="minibatch",
                update_type=3,
                alpha=alpha,
                beta=beta,
            )
            predictions = (
                np.argmax(softmax(np.dot(X_test, weights), axis=1), axis=1) + 1
            )
            print(predictions)
            write_to_file(predictions, output_file)
            write_to_file(weights.flatten(), weight_file)

    elif mode == "c":
        pass

    elif mode == "d":
        pass
