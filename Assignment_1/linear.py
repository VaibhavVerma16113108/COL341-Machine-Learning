import numpy as np
import scipy as sp
import pandas as pd
import sys
from sklearn import preprocessing

FEATURES = [
    "APR Medical Surgical Description",
    "Health Service Area APR Medical Surgical Description",
    "Hospital County APR DRG Code",
    "Operating Certificate Number Length of Stay",
    "Operating Certificate Number CCS Diagnosis Code",
    "Operating Certificate Number APR Medical Surgical Description",
    "Facility Name Length of Stay",
    "Gender APR Medical Surgical Description",
    "Length of Stay^2",
    "Length of Stay Type of Admission",
    "Length of Stay CCS Diagnosis Code",
    "Length of Stay APR DRG Code",
    "Length of Stay APR Severity of Illness Code",
    "Length of Stay APR Medical Surgical Description",
    "CCS Diagnosis Code APR DRG Code",
    "CCS Diagnosis Code APR MDC Code",
    "CCS Diagnosis Code APR MDC Description",
    "CCS Diagnosis Code APR Severity of Illness Description",
    "CCS Diagnosis Code APR Risk of Mortality",
    "APR DRG Code APR Risk of Mortality",
    "APR DRG Code APR Medical Surgical Description",
    "APR MDC Description Emergency Department Indicator",
    "APR Severity of Illness Code APR Medical Surgical Description",
    "Emergency Department Indicator^2",
]


def preprocess_dataset(dataset_file, type="train"):
    data = pd.read_csv(dataset_file)
    data.drop("Unnamed: 0", inplace=True, axis=1)
    all_ones = [1 for _ in range(data.shape[0])]
    data.insert(loc=0, column="Ones", value=all_ones)

    if type == "train":
        X, y = data.iloc[:, : data.shape[1] - 1], data.iloc[:, -1:]
        return X, y
    else:
        return data


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
            error += np.sum(np.square(y_pred - y_test)) / np.sum(np.square(y_test))
        if error < min_error:
            min_error = error
            optimal_lambda = reg_param
    return optimal_lambda


def extract_features(X, features=FEATURES):
    X.drop("Birth Weight", axis=1, inplace=True)
    ENCODING_THRESHOLD = 10
    columns_to_encode = [
        col for col in X.columns if len(X[col].unique()) <= ENCODING_THRESHOLD
    ]
    poly = preprocessing.PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_expanded = pd.DataFrame(
        X_poly, columns=poly.get_feature_names(input_features=X.columns)
    )
    X_expanded = X_expanded[features]
    return X_expanded


def predict(test_file, weights, features=None):
    """
    This function predicts the output for the test dataset.
    """
    data = preprocess_dataset(test_file, type="test")
    if features is not None:
        data = extract_features(data)
        data = data.loc[:, features]
    X_test = data.to_numpy()
    y_pred = X_test.dot(weights)

    return y_pred


def predict_2(train_file, weights):
    """
    This function predicts the output for the train dataset.
    """
    X_train, y_train = preprocess_dataset(train_file)
    y_pred = X_train.dot(weights)

    return y_pred


def r_score(y_test, y_pred):
    """
    This function returns the R-squared value.
    """
    mean_y = np.mean(y_test)
    numerator = np.sum((y_test - y_pred) ** 2)
    denominator = np.sum((y_test - mean_y) ** 2)
    return 1 - (numerator / denominator)


def write_to_file(arr, filename, write_mode="Array"):
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
        X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
        weights = normal_equation(X_train, y_train)
        predictions = predict(test_data_file, weights)
        write_to_file(predictions, output_file)
        write_to_file(weights, weight_file)

    elif mode == "b":
        regularization_file, output_file, weight_file, best_param_file = (
            sys.argv[4],
            sys.argv[5],
            sys.argv[6],
            sys.argv[7],
        )
        lambda_ = np.loadtxt(regularization_file, delimiter=",")
        X_train, y_train = preprocess_dataset(train_data_file)
        X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
        optimal_lambda = k_folds_cross_validation(X_train, y_train, 10, lambda_)
        optimal_weights = ridge_regression(X_train, y_train, optimal_lambda)
        predictions = predict(test_data_file, optimal_weights)
        write_to_file(predictions, output_file)
        write_to_file(optimal_weights, weight_file)
        write_to_file(optimal_lambda, best_param_file, write_mode="Scalar")

    elif mode == "c":
        output_file = sys.argv[4]
        X_train, y_train = preprocess_dataset(train_data_file)
        X_train = extract_features(X_train).to_numpy()
        weights = normal_equation(X_train, y_train)
        predictions = predict(test_data_file, weights, features=FEATURES)
        write_to_file(predictions, output_file)
