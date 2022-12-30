import xgboost
from sklearn import preprocessing, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from Estimators import *
from sklearn.metrics import f1_score, roc_auc_score, precision_score, classification_report, precision_recall_curve, \
    confusion_matrix
import numpy as np
import matplotlib as plt
import itertools


class BestResult:
    def __init__(self, threshold, fscore, precision, recall):
        self.threshold = threshold
        self.fscore = fscore
        self.precission = precision
        self.recall = recall


class ModelAnalyzer:
    def __init__(self, model_name, model, continuous_columns, categorical_columns):
        self.model_name = model_name
        self.model = model
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns
        self.best_result: BestResult = BestResult(0, 0, 0, 0)
        self.y_pred = None
        self.y_test = None

    def make_feature_union(self) -> FeatureUnion:
        final_transformers = list()
        for col in self.continuous_columns:
            transformer = Pipeline([
                ('numeric', NumberSelector(key=col)),
                ('minmax', MinMaxScaler(key=col)),
            ])
            final_transformers.append((col, transformer))
        for col in self.categorical_columns:
            transformer = Pipeline([
                ('selector', FeatureSelector(column=col)),
                ('ohe', OHEEncoder(key=col))
            ])
            final_transformers.append((col, transformer))
        return FeatureUnion(final_transformers)

    def analyze_model(self, X_train, y_train, X_test, y_test):
        self.y_test = y_test
        feats = self.make_feature_union()
        print()
        pipeline = Pipeline([
            ('features', feats),
            ('classifier', self.model),
        ])
        pipeline.fit(X_train, y_train)
        self.y_pred = pipeline.predict_proba(X_test)[:, 1]

        precision, recall, threshold = precision_recall_curve(y_test, self.y_pred)
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix = np.argmax(fscore)
        self.best_result = BestResult(
            fscore=fscore[ix],
            threshold=threshold[ix],
            precision=precision[ix],
            recall=recall[ix]
        )

    def report(self):
        print("-"*80)
        print(self.model_name)
        print(f'Прогноз (первые 10): {self.y_pred[:10]}')
        print()
        print(f'Best: Threshold={self.best_result.threshold},'
              f' F-Score={self.best_result.fscore},'
              f' Precision={self.best_result.precission},'
              f' Recall={self.best_result.recall}')
        print()
        print(f'Доля людей, не желающих уйти в отток, '
              f'но распознанных, как ушедшие в отток: '
              f'{1 - self.best_result.precission}')
        print(f'Доля людей, желающих уйти в отток, '
              f'но распознанных, как не желающие уйти в отток: '
              f'{1 - self.best_result.recall}')
        print("-"*80)

