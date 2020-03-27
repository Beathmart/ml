from ml.probability import *
from ml.utils import *
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment',None)
from ml.utils import pd_to_np_1d


def best_split_entropy(data, features, target):
    """
        We want to select out the best feature such that it splits the data best based on your measurement(IG/accuracy)
        Input: (Pandas DataFrame)data
               (List of String) features  candidates we can choose feature from
               (String) target  the target name we shoot for.

        Output: (String) the best feature
    """
    best_feature = None
    best_info_gain = -1.0
    # entropy_target = information_entropy(data[target])

    for feature in features:
        info_gain = information_gain(data[feature], data[target])
        # print('the gain of feature %s is: %f'%(feature, info_gain))
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    return best_feature

def best_split_gini(data, features, target):
    best_feature = None
    best_gini_index = 1.0

    for feature in features:
        gini_index = gini_impurity(data[feature], data[target])
        if gini_index < best_gini_index:
            best_gini_index = gini_index
            best_feature = feature
    return best_feature

def split_continuous_to_binary(data, feature, target):
    x = data[feature]
    y = data[target]

    x_sorted = x.sort_values().values
    split_points = []
    quantiles = [.2, .4, .6, .8]
    for q in quantiles:
        split_points.append(np.quantile(x_sorted, q))

    # for i in range(len(x_sorted) - 1):
    #     split_points.append((x_sorted[i] + x_sorted[i+1]) / 2)

    best_gain = -1.0
    best_split_point = None
    best_split_data = None
    for split_point in split_points:
        x_split = x.apply(lambda v: 0 if v < split_point else 1)
        info_gain = information_gain(x_split, y)
        # print(split_point, info_gain)
        if info_gain > best_gain:
            best_gain = info_gain
            best_split_point = split_point
            best_split_data = x_split
    return best_split_data, best_gain, best_split_point

def bianry_split_dataset(data, feature, value):
    left = data[data[feature] <= value]
    right = data[data[feature] > value]
    return left, right

class DecisionTree:
    '''
    implemented by dictionary for binary classification
    still have many problems to be solved
    '''
    def __init__(self):
        pass

    @print_run_time
    def fit(self, X, y):
        '''
        :param X: (pd dataframe)
        :param y: (pd dataframe/ pd series)
        :return: (dict)
        '''
        data = pd.concat([X, y], axis=1)
        features = data.columns[:-1]
        target = data.columns[-1]

        ## transfer continous value to binary
        data = self._process_continous(data)

        ### original unique values of every feature
        self.unique_values = {}
        for feature in features:
            self.unique_values[feature] = np.unique(data[feature])

        self.tree = self.create_tree(data, features, target)


    def _process_continous(self, data):
        labels = [label for label in data]
        X_labels = labels[:-1]
        y_label = labels[-1]

        self.data_processed = dict()
        for x in X_labels:
            if data[x].dtypes == 'float':
                split = split_continuous_to_binary(data, x, y_label)
                data[x] = split[0]
                self.data_processed[x] = split[-1]
        return data

    def predict(self, X):
        '''

        :param X: (pd dataframe/pd series) mxn
        :return: (int) class 0 or 1 mx1
        '''

        X = X.copy()
        for feat in self.data_processed.keys():
            X[feat] = np.where(X[feat] < self.data_processed[feat], 0, 1)

        prediction = X.apply(lambda row: self.predict_single_data(self.tree, row), axis=1)
        return prediction

    def predict_single_data(self, tree, x):
        '''
        :param x: (pd dataframe/pd series)
        :return: (int) class 0 or 1
        '''

        first_feat = [key for key in tree.keys()][0]
        second_dict = tree[first_feat]

        # class_label = None
        for key in second_dict.keys():
            if x[first_feat] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = self.predict_single_data(second_dict[key], x)
                else:
                    class_label = second_dict[key]
        # problem if leaf does not include all categories in a feature. ***to be solved
        # if class_label == None:
        #     print(first_feat)
        #     print(tree[first_feat])
        #     raise ValueError()
        return class_label


    def score(self, X_test, y_test):
        '''

        :param X_test:
        :param y_test:
        :return:
        '''
        m = len(X_test)
        pred = self.predict(X_test)
        acc = 0
        pred = pred.values
        y_test = y_test.values
        for i in range(m):
            if pred[i] == y_test[i]:
                acc += 1
        return acc / m


    def create_tree(self, data, features, target):
        if len(np.unique(data[target])) == 1:
            return np.unique(data[target])[0]
        if len(features) == 0:
            return self.leaf_class(data[target])

        split_feature = best_split_entropy(data, features, target)
        unique_split_feature_values = np.unique(data[split_feature])
        if len(unique_split_feature_values) < len(self.unique_values[split_feature]):
            return self.leaf_class(data[target])

        tree = {split_feature:{}}
        features = features.drop(split_feature)

        for value in unique_split_feature_values:
            sub_dataset = data[data[split_feature]==value]
            tree[split_feature][value] = self.create_tree(sub_dataset, features, target)
        return tree

    def leaf_class(self, target_values):
        '''
        :param target_values:  (np array/pd dataframe/pd series) 1d target values
        :return: (int) 1 or 0
        '''
        positives = len(target_values[target_values == 1])
        negatives = len(target_values[target_values == 0])

        return 1 if positives > negatives else 0


class TreeNode:
    def __init__(self, value=None, trueBranch=None, falseBranch=None, results=None, col=None, summary=None, data=None, probas=None):
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results
        self.col = col
        self.summary = summary
        self.data = data
        self.probas = probas

class DecisionTreeClassifier:
    '''
    only for binary classification
    '''
    def __init__(self):
        pass

    @print_run_time
    def fit(self, X, y, sample_weight=None):
        data = pd.concat([X, y], axis=1)
        features = data.columns[:-1]
        target = data.columns[-1]

        self.features = features
        self.target = target

        self.tree = self.create_tree(data, features, target, sample_weight)

    def predict(self, X):
        predictions = X.apply(lambda row: self.predict_single_data(row, self.tree), axis=1)
        return predictions

    def predict_single_data(self, x, tree):
        if tree.results != None:
            # print('************')
            return tree.results
        else:
            branch = None
            v = x[tree.col]
            # print(tree.col)
            # print(v, tree.value)
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
            else:
                if v == tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
            return self.predict_single_data(x, branch)

    def predict_proba(self, X):
        if len(X) == 1:
            if isinstance(X, np.ndarray):
                X = pd.Series(X[0], index=self.features)

            predictions = self.predict_proba_single_data(X, self.tree)
            predictions = predictions.reshape(1, -1)
        else:
            predictions = X.apply(lambda row: self.predict_proba_single_data(row, self.tree), axis=1)
            predictions = np.array(predictions.tolist())
        return predictions

    def predict_proba_single_data(self, x, tree):
        if tree.probas is not None:
            return tree.probas
        else:
            branch = None
            v = x[tree.col]
            # print(tree.col)
            # print(v, tree.value)
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
            else:
                if v == tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
            return self.predict_proba_single_data(x, branch)


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

    def create_tree(self, data, features, target, sample_weight=None):
        # current_gain = gini(data[target]) ##
        # g = gini(data[target]) ##
        m = len(data)

        best_gain = 1.1 ##
        best_value = None
        best_set = None

        if sample_weight is not None:
            data = pd.concat([data, pd.Series(sample_weight, name='sample_weight')], axis=1)

        for feature in features:
            values = np.unique(data[feature])
            for value in values:
                left, right = self.split_data(data, feature, value)
                if sample_weight is None:
                    p = len(left) / m
                else:
                    p = np.sum(left['sample_weight']) / np.sum(data['sample_weight'])
                gain =  p * gini(left[target]) + (1 - p) * gini(right[target]) ##
                if gain < best_gain:  ##
                    best_gain = gain
                    best_value = (feature, value)
                    best_set = (left, right)
        summary = {'gini': '%.3f'%best_gain, 'sample': '%d'%m} ##

        # stop condition
        if best_gain > 0:
            left_branch = self.create_tree(best_set[0], features, target)
            right_branch = self.create_tree(best_set[1], features, target)
            return TreeNode(col=best_value[0], value=best_value[1], trueBranch=left_branch, falseBranch=right_branch, summary=summary)
        else:
            uniques = self.unique_count(data, target)
            if len(uniques) == 1:
                uniques[1-list(uniques.keys())[0]] = 0
            cls = sorted(uniques.items(), key=lambda item: item[1])[-1][0]

            uniques_list = sorted(uniques.items(), key=lambda item: item[0])
            uniques_list = np.array(uniques_list)
            probas = uniques_list[:, 1] / np.sum(uniques_list[:, 1])
            # print(probas)
            # probas = probas.tolist()

            # majority = -1.0
            # cls = None
            # for key in uniques.keys():
            #     if uniques[key] > majority:
            #         majority = uniques[key]
            #         cls = key
            # print(cls)
            return TreeNode(results=cls, probas=probas, summary=summary, data=data, col=target) ##

    def unique_count(self, data, target):
        results = {}
        for d in data[target]:
            if d not in results:
                results.setdefault(d, 1)
            else:
                results[d] += 1
        return results

    def split_data(self, data, feature, value):

        if isinstance(value, int) or isinstance(value, float):
            left = data[data[feature]>=value]
            right = data[data[feature]<value]
        else:
            left = data[data[feature]==value]
            right = data[data[feature]!=value]

        return left, right

class RegNode:
    def __init__(self, value=None, left=None, right=None, results=None, col=None, data=None):
        self.value = value
        self.left = left
        self.right = right
        self.results = results
        self.col = col
        self.data = data

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_error_gain=1e-5, min_leaf_samples=4):
        self.max_depth = max_depth
        self.min_error_gain = min_error_gain
        self.min_leaf_samples = min_leaf_samples

    @print_run_time
    def fit(self, X, y):
        data = pd.concat([X, y], axis=1)
        features = data.columns[:-1]
        target = data.columns[-1]

        self.features = features
        self.target = target

        current_error = self._sse(data, target)  ## mse, sse
        self.tree = self.create_tree(data, features, target, current_error, current_depth=0)

    def predict(self, X):
        '''

        :param X:
        :return: (pd Series)
        '''
        predictions = X.apply(lambda row: self.predict_single_data(row, self.tree), axis=1)
        return predictions

    def predict_single_data(self, x, tree):
        if tree.results != None:
            return tree.results
        else:
            branch = None
            v = x[tree.col]
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.left
                else:
                    branch = tree.right
            else:
                if v == tree.value:
                    branch = tree.left
                else:
                    branch = tree.right
            return self.predict_single_data(x, branch)

    def score(self, X_test, y_test):
        from ml.metrics import r2_score
        # from sklearn.metrics import r2_score
        # from sklearn.metrics import mean_squared_error as MSE
        # from ml.metrics import mean_squared_error
        preds = self.predict(X_test)
        return r2_score(y_test, preds)

    def _mean_squared_error(self, y, h):
        y = pd_to_np_1d(y)
        h = pd_to_np_1d(h)
        m = len(y)
        mse = (1.0 / m) * np.sum((y - h) ** 2)
        return mse

    def create_tree(self, data, features, target, current_error, current_depth):
        if self.max_depth != None:
            if current_depth >= self.max_depth:
                return self._leaf_node(data, target)

        if len(data) < self.min_leaf_samples:
            return self._leaf_node(data, target)

        best_error, best_set, best_feature, best_value = self._best_feature(data, features, target)

        left = best_set[0]
        right = best_set[1]

        if len(left) == len(data):
            return self._leaf_node(left, target)
        if len(right) == len(data):
            return self._leaf_node(right, target)

        if (current_error - best_error) <= self.min_leaf_samples:
            return self._leaf_node(data, target)

        left_branch = self.create_tree(left, features, target, best_error, current_depth+1)
        right_branch = self.create_tree(right, features, target, best_error, current_depth+1)
        return RegNode(col=best_feature, value=best_value, left=left_branch, right=right_branch)

        # if (current_error - best_mse) > self.min_error_gain:
        #     left_branch = self.create_tree(best_set[0], features, target, best_mse, current_depth+1)
        #     right_branch = self.create_tree(best_set[1], features, target, best_mse, current_depth+1)
        #     return RegNode(col=best_feature, value=best_value, left=left_branch, right=right_branch)
        # else:
        #     return self._leaf_node(data, target)

    def _leaf_node(self, data, target):
        results = self._mean(data, target)
        return RegNode(results=results, col=target, data=data)

    def _split_points(self, data, feature):
        values = np.unique(data[feature])
        m = len(values)
        if m < 2:
            return values

        if values.dtype == 'float' or values.dtype == 'int':
            values = sorted(values)
            points = []
            for i in range(m-1):
                point = (values[i] + values[i+1]) / 2
                points.append(point)
            n = len(points)
            points = np.random.choice(points, int(np.log2(n) + 1), replace=False)
            return points
        return values

    def _split_data(self, data, feature, value):
        if isinstance(value, int) or isinstance(value, float):
            left = data[data[feature]>=value]
            right = data[data[feature]<value]
        else:
            left = data[data[feature]==value]
            right = data[data[feature]!=value]
        return left, right

    def _best_feature(self, data, features, target):

        best_error = float('inf')
        best_set = None
        best_feature = None
        best_value = None

        for feature in features:
            values = self._split_points(data, feature)
            # values = np.unique(data[feature])
            for value in values:
                left, right = self._split_data(data, feature, value)
                error = self._sse(left, target) + self._sse(right, target)  ## mse, sse
                if error < best_error:
                    best_error = error
                    best_set = (left, right)
                    best_feature = feature
                    best_value = value

        return best_error, best_set, best_feature, best_value

    def _mse(self, data, target):
        m = len(data)
        if m <= 0:
            return 0

        mean = self._mean(data, target)
        mse = (1.0 / m) * ((mean - data[target]) ** 2).sum()
        return mse

    def _sse(self, data, target):
        return np.var(data[target]) * len(data)

    def _mean(self, data, target):
        return np.mean(data[target])

if __name__ == '__main__':
    ## ----------***********----------
    ## simple test
    # data = pd.DataFrame(
    #                     {
    #                         'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    #                         'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    #                         'Humidity': [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    #                         'Wind': [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    #                         'Play': [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]
    #                     }
    #                     )
    # # print(best_split_entropy(data, ['Humidity', 'Wind'], 'Play'))
    # X = data.drop(['Play'], axis=1)
    # y = data['Play']
    #
    # from sklearn.model_selection import train_test_split
    # X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=3)
    # print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)
    #
    # model = DecisionTree()
    # model.fit(X_train, y_train)
    # print(model.tree)
    # predictions = model.predict(X_vali)
    # print(model.score(X_vali, y_vali))
    #
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.preprocessing import OneHotEncoder
    # dummies_outlook = pd.get_dummies(data['Outlook'], prefix='Outlook')
    # dummies_temperature = pd.get_dummies(data['Temperature'], prefix='Temperature')
    # df = pd.concat([data, dummies_outlook, dummies_temperature], axis=1)
    # df.drop(['Outlook', 'Temperature'], axis=1, inplace=True)
    # print(df.head())
    #
    # X = df.drop(['Play'], axis=1)
    # y = df['Play']
    # X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=3)
    #
    # sklearn_dt = DecisionTreeClassifier()
    # sklearn_dt.fit(X_train, y_train)
    # print(sklearn_dt.score(X_vali, y_vali))

    ## ------------------****************------------------
    ## sklearn dataset
    # wm = pd.DataFrame(
    #                    {
    #                      'Density': [0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343,
    #                                0.639, 0.657, 0.360, 0.593, 0.719],
    #                      'Target': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #                    }
    #                   )
    #
    # _, g, t = split_continuous_to_binary(wm, 'Density', 'Target')
    # print('best gain: %.3f'%g, '\nbest split point: ', t)
    from sklearn.datasets import make_blobs, make_classification
    # X, y = make_blobs(n_samples=203, centers=2, n_features=5, random_state=14)
    X, y = make_classification(n_samples=1001, n_features=3, n_redundant=0, n_informative=3, n_clusters_per_class=1, random_state=15)
    df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1), columns=['f1', 'f2', 'f3', 'y'])
    df['y'] = df['y'].astype('int')
    X = df.drop('y', axis=1)
    y = df['y']

    from sklearn.model_selection import train_test_split
    X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=.2, random_state=15)
    # print(X_train[:5])
    # print(X_vali[:5])

    import time
    from sklearn.tree import  DecisionTreeClassifier as sklearn_tree
    sklearn_dt = sklearn_tree()
    start = time.time()
    sklearn_dt.fit(X_train, y_train)
    print('sklearn model run time is %.2f'%(time.time()-start))
    print('sklearn model score: ', sklearn_dt.score(X_vali, y_vali))
    print('\n')

    MyDecisionTree = DecisionTree()
    MyDecisionTree.fit(X_train, y_train)
    print(MyDecisionTree.tree)
    print('my model score: ', MyDecisionTree.score(X_vali, y_vali))
    print('\n')

    cart = DecisionTreeClassifier()
    cart.fit(X_train, y_train)
    # print(np.unique(cart.predict(X_vali)))
    # print(cart.predict(X_vali[:20]))
    # print(y_vali[:20])
    # print('cart score in TrainSet: ', cart.score(X_train, y_train))
    # print(type(cart.predict(X_vali)))
    # print(cart.predict(X_vali).shape)
    print('cart score in ValiSet: ', cart.score(X_vali, y_vali))