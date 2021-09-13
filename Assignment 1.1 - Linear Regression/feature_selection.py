import sys
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model


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


def create_poly_features(X_train, degree):
    """
    Create polynomial features up to degree 'degree'
    :param X_train: Training set
    :param degree: Degree of polynomial
    :return: X_train with polynomial features
    """
    poly = preprocessing.PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    X_expanded = pd.DataFrame(
        X_poly, columns=poly.get_feature_names(input_features=X_train.columns)
    )
    return X_expanded


def one_hot_encode(X, encoding_threshold=15):
    """
    One-hot encode categorical features
    X: Training set
    encoding_threshold: Threshold for categorical features to be encoded
    :return: X with one-hot encoded features
    """
    columns_to_encode = []
    for col in X.columns:
        if len(X[col].unique()) <= encoding_threshold:
            columns_to_encode.append(col)
    columns_to_encode.remove("Ones")
    for col in columns_to_encode:
        dummies = pd.get_dummies(df.loc[:, col], prefix=col, drop_first=False)
        X = pd.concat([X, dummies], axis=1)
    return X


def drop_features(X, y, threshold=0.01):
    """
    Drop specified features
    :param X: Training set
    :param y: Target
    :param threshold: Correlation threshold
    :return: X with dropped features
    """
    # Drop features whose correlation with target is below threshold
    # cols = [col for col in X.columns if abs(X[col].corr(y["Total Costs"])) < threshold]

    # Drop 'Birth Weight' as it has a lot of missing values. Drop repeated features.
    cols = [
        "Birth Weight",
        "APR MDC Code",
        "CCS Diagnosis Code",
        "CCS Procedure Code",
        "APR Severity of Illness Code",
    ]
    X.drop(cols, axis=1, inplace=True)


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
    optimal_lambda = 0
    best_r_score = 0

    for reg_param in lambda_:
        curr_r_score = 0
        for iteration in range(k):
            # Split the data into k folds
            X_train, X_test, y_train, y_test = train_test_split(X, y, k, iteration)
            print("Split done...")
            # Fit the model
            lasso = linear_model.LassoLars(alpha=reg_param, max_iter=100)
            lasso.fit(X_train, y_train)
            # Predict the labels
            curr_r_score += lasso.score(X_test, y_test)
        curr_r_score /= k
        if curr_r_score > best_r_score:
            best_r_score = curr_r_score
            optimal_lambda = reg_param
        print("r score: ", curr_r_score, " and lambda: ", reg_param)
    return optimal_lambda, best_r_score