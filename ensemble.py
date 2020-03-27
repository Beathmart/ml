import numpy as np
import copy
from ml.utils import print_run_time
import pandas as pd

class AdaBoostClassifier:
    def __init__(self, base_model, n_estimators=50):
        self.base_model = base_model
        self.n_estimators = n_estimators


    def fit(self, X, y):

        self.models, self.model_weights = self.fit_helper(X, y)

    def fit_helper(self, X, y):
        '''

        :param X: (pd DataFrame)
        :param y: (pd DataFrame/Series)
        :return:
        '''
        m = len(y) # num of samples
        sample_weights = np.ones((m,)) * 1.0 / m # shape (m,)
        models = []
        model_weights = []

        for t in range(self.n_estimators):
            model = copy.deepcopy(self.base_model)
            model.fit(X, y, sample_weight=sample_weights)

            y_values = y.values
            pred = model.predict(X) # shape (m,) type (np array/pd Series)
            error = np.sum((pred != y_values) * sample_weights) / np.sum(sample_weights)
            if error > 0.5:
                break
            alpha = 0.5 * np.log((1 - error) / error)
            models.append(model)
            model_weights.append(alpha)

            if not isinstance(pred, np.ndarray):
                pred = pred.values
            update = np.exp(-alpha * np.array([1 if pred[i]==y_values[i] else -1 for i in range(m)])) # (m,)
            sample_weights = sample_weights * update # (m,)
            sample_weights /= np.sum(sample_weights) # normalize

        return models, model_weights


    def predict(self, X):
        predictions = X.apply(lambda row: self.predict_single_data(row), axis=1)
        return predictions


    def predict_single_data(self, x):
        x = x.values.reshape(1, -1) # (1, -1) np array

        pred_probas = []
        for model in self.models:
            pred_proba = model.predict_proba(x) # (1, n_classes)
            pred_probas.append(pred_proba[0][1])

        pred_probas = np.array(pred_probas) # (T,)
        model_weights = np.array(self.model_weights) # (T,)
        H = (1 if np.sum((pred_probas * model_weights)) > np.sum((1 - pred_probas) * model_weights) else 0)
        return H


    def score(self, X_test, y_test):
        m = len(X_test)
        pred = self.predict(X_test)
        acc = 0
        pred = pred.values
        y_test = y_test.values
        for i in range(m):
            if pred[i] == y_test[i]:
                acc += 1

        return acc / m

class GradientBoostingRegressor:
    def __init__(self, learning_rate=0.1, n_estimators=50, max_depth=3, min_leaf_samples=4):
        self.trees = None
        self.learning_rate = learning_rate
        self.init_val = None

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_leaf_samples = min_leaf_samples

    def fit(self, X, y):
        from ml.tree import DecisionTreeRegressor
        if isinstance(y, pd.core.frame.DataFrame):
            y = y[y.columns[-1]]

        if isinstance(y, np.ndarray):
            raise ValueError('data (y) must be pandas DataFrame/Series')

        self.init_val = self._get_init_val(y)
        m = len(y)
        h = np.array([self.init_val] * m)
        residuals = self._get_residual(y, h)

        self.trees = []

        for k in range(self.n_estimators):
            residuals_sub = residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_leaf_samples=self.min_leaf_samples)
            tree.fit(X, residuals_sub)

            h = h + self.learning_rate * tree.predict(X)#.values.reshape(-1)
            residuals = self._get_residual(y, h)
            self.trees.append(tree)

    def predict(self, X):
        predictions = self.init_val + np.sum(self.learning_rate * tree.predict(X) for tree in self.trees)
        return predictions

    def score(self, X_test, y_test):
        from ml.metrics import r2_score
        preds = self.predict(X_test)
        return r2_score(y_test, preds)

    def _get_init_val(self, y):
        return np.mean(y)

    def _get_residual(self, y, h):
        residual = y - h
        return residual


if __name__ == '__main__':
    pass