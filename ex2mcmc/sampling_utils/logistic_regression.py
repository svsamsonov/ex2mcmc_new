import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn import datasets, preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


def filter_outliers(X, random_state=42):
    clf = IsolationForest(random_state=random_state)
    # clf = LocalOutlierFactor(n_neighbors=20) #, contamination=0.1)
    # use fit_predict to compute the predicted labels of the training samples
    # (when LOF is used for outlier detection, the estimator has no predict,
    # decision_function and score_samples methods).
    y_pred = clf.fit_predict(X)
    X = X[y_pred == 1]

    return y_pred


def standartize(X_train, X_test, intercept=True):
    """Whitens noise structure, covariates updated"""
    X_train = copy.deepcopy(X_train)
    X_test = copy.deepcopy(X_test)
    # if intercept: #adds intercept term
    #     X_train = np.concatenate((np.ones(X_train.shape[0]).reshape(X_train.shape[0],1),X_train),axis=1)
    #     X_test = np.concatenate((np.ones(X_test.shape[0]).reshape(X_test.shape[0],1),X_test),axis=1)
    # d = X_train.shape[1]

    # Centering the covariates
    means = np.mean(X_train, axis=0)
    if intercept:  # do not subtract the mean from the bias term
        means[0] = 0.0
    # Normalizing the covariates
    X_train -= means
    Cov_matr = np.dot(X_train.T, X_train)
    U, S, V_T = np.linalg.svd(Cov_matr, compute_uv=True)
    # Sigma_half = U @ np.diag(np.sqrt(S)) @ V_T
    Sigma_minus_half = U @ np.diag(1.0 / np.sqrt(S)) @ V_T
    X_train = X_train @ Sigma_minus_half
    # The same for test sample
    X_test = (X_test - means) @ Sigma_minus_half
    return X_train, X_test


def split(X, y, c1, c2):
    X = X.astype("float32")
    y = y.astype("int")

    # Select the classes to classify for binary logistic regression
    i1 = np.where((y == c1) | (y == c2))
    y = y[i1]
    X = X[i1]

    # Replace labels of the current class with 1 and other labels with 0
    i1 = np.where(y == c1)
    i2 = np.where(y == c2)
    y[i1] = 1
    y[i2] = 0

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    return x_train, x_test, y_train, y_test


def preprocess(x_train, x_test, y_train, y_test, normalize=False):
    # Making sure that the values are float so that we can get decimal points
    # after division

    outliers_pred = filter_outliers(x_train)
    print(
        f"Original size: {len(x_train)}, outliers size: {sum(outliers_pred == -1)}",
    )

    x_train = x_train[outliers_pred == 1]
    y_train = y_train[outliers_pred == 1]

    min_max_scaler = preprocessing.MinMaxScaler()

    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)

    x_train = np.concatenate(
        (np.ones(x_train.shape[0]).reshape(x_train.shape[0], 1), x_train),
        axis=1,
    )
    x_test = np.concatenate(
        (np.ones(x_test.shape[0]).reshape(x_test.shape[0], 1), x_test),
        axis=1,
    )

    if normalize:
        x_train, x_test = standartize(x_train, x_test, intercept=True)

    # Return objects of interest
    return x_train, x_test, y_train, y_test


def import_covertype(c1=1, c2=2, n_data=500000):
    dataset = datasets.fetch_covtype()
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X = min_max_scaler.fit_transform(dataset.data)
    X = dataset.data
    y = dataset.target
    # x_train = X[0:n_data,:]
    # x_test = X[n_data:,:]
    # y_train = dataset.target[0:n_data]
    # y_test = dataset.target[n_data:]
    x_train, x_test, y_train, y_test = split(X, y, c1, c2)

    return preprocess(x_train, x_test, y_train, y_test, normalize=False)


def import_breast(c1=0, c2=1, n_data=569):
    dataset = datasets.load_breast_cancer()
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X = min_max_scaler.fit_transform(dataset.data)
    X = dataset.data
    y = dataset.target

    # X = dataset.data
    # x_train = X[0:n_data,:]
    # x_test = X[n_data:,:]
    # y_train = dataset.target[0:n_data]
    # y_test = dataset.target[n_data:]
    x_train, x_test, y_train, y_test = split(X, y, c1, c2)

    return preprocess(x_train, x_test, y_train, y_test)


def import_csv_dataset(path, c1=0, c2=1):
    dataset = pd.read_csv(path, delimiter=",")
    dataset = dataset.to_numpy()
    X = dataset[:, :-1]
    y = dataset[:, -1]
    x_train, x_test, y_train, y_test = split(X, y, c1, c2)

    return preprocess(x_train, x_test, y_train, y_test)


def import_digits(c1=5, c2=6):
    dataset = datasets.load_digits()
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X = min_max_scaler.fit_transform(dataset.data)
    X = dataset.data
    y = dataset.target

    # X = dataset.data
    # x_train = X[0:n_data,:]
    # x_test = X[n_data:,:]
    # y_train = dataset.target[0:n_data]
    # y_test = dataset.target[n_data:]
    x_train, x_test, y_train, y_test = split(X, y, c1, c2)

    return preprocess(x_train, x_test, y_train, y_test)


@dataclass
class ClassificationDataset:
    x_train: torch.Tensor
    x_test: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor

    @property
    def d(self):
        return self.x_train.shape[1]

    @property
    def n(self):
        self.y_train.shape[0]


class ClassificationDatasetFactory:
    def __init__(self, data_root=None, device="cpu", **kwargs):
        self._instance = None
        self.data_root = data_root
        self.device = device

    def get_dataset(self, name: str, **kwargs) -> ClassificationDataset:
        if name == "covertype":
            x_train, x_test, y_train, y_test = import_covertype(**kwargs)
        elif name == "breast":
            x_train, x_test, y_train, y_test = import_breast(**kwargs)
        elif name == "digits":
            x_train, x_test, y_train, y_test = import_digits(**kwargs)
        else:
            # csv
            if not name.endswith(".csv"):
                name = f"{name}.csv"
            path = Path(self.data_root, name)
            x_train, x_test, y_train, y_test = import_csv_dataset(
                path,
                **kwargs,
            )

        x_train = torch.tensor(
            x_train,
            dtype=torch.float32,
            device=self.device,
        )
        x_test = torch.tensor(x_test, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(
            y_train,
            dtype=torch.float32,
            device=self.device,
        )
        y_test = torch.tensor(y_test, dtype=torch.float32, device=self.device)

        return ClassificationDataset(x_train, x_test, y_train, y_test)


# Import data

"""
w, v = np.linalg.eig(np.dot(X.T,X))
w = np.max(w.real)
M = w * (1/4) + tau
gamma = 0.01/M

# Initialization
d = np.size(X,1)
n = np.size(y)
tau = 1 # regularization parameter
"""
