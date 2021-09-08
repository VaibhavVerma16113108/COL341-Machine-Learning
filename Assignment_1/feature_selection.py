import sys
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.model_selection import KFold, train_test_split


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
    Drop features with a missing rate above threshold
    :param X: Training set
    :param y: Target
    :param threshold: Missing rate threshold
    :return: X with dropped features
    """
    cols = [col for col in X.columns if abs(X[col].corr(y["Total Costs"])) < threshold]
    cols += [
        "Birth Weight",
        "APR MDC Code",
        "CCS Diagnosis Code",
        "CCS Procedure Code",
        "APR Severity of Illness Code",
    ]
    X.drop(cols, axis=1, inplace=True)


def cross_validation(X, y, lambda_):
    """
    Returns the best regularization parameter using cross-validation.
    """
    optimal_alpha = 0
    best_r_score = 0
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=False
    )
    for alpha in lambda_:
        r_score = 0
        for train_index, test_index in kf.split(X_train):
            X_train_cv, X_test_cv = X_train.loc[train_index], X_train.loc[test_index]
            y_train_cv, y_test_cv = y_train.loc[train_index], y_train.loc[test_index]
            lr = linear_model.LassoLars(alpha=alpha, max_iter=1000)
            lr.fit(X_train_cv, y_train_cv)
            r_score += lr.score(X_test_cv, y_test_cv)
        r_score /= 10
        if r_score > best_r_score:
            best_r_score = r_score
            optimal_alpha = alpha
    return optimal_alpha