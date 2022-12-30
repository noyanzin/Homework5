from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        x = X[[self.key]]
        return X[[self.key]]


class OHEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        self.columns = []

    def fit(self, X, y=None):
        self.columns = [col for col in pd.get_dummies(X, prefix=self.key).columns]
        return self

    def transform(self, X):
        X = pd.get_dummies(X, prefix=self.key)
        test_columns = [col for col in X.columns]
        for col_ in self.columns:
            if col_ not in test_columns:
                X[col_] = 0
        return X[self.columns]


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """
    MinMaxScaler
    """

    def __init__(self, key):
        self.key = key
        self.min: float = None
        self.max: float = None

    def fit(self, X: pd.DataFrame, y=None):
        self.min = X[self.key].min()
        self.max = X[self.key].max()
        return self

    def transform(self, X):
        x = X[[self.key]]
        x_out = (x - self.min) / (self.max - self.min)
        return x_out
