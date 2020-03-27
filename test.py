import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ml import *

'''
test anything here
'''

def classification():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=3, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                               n_classes=2, random_state=15, class_sep=1.0)
    df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1), columns=['f1', 'f2', 'f3', 'y'])
    df['y'] = df['y'].astype('int')
    X = df.drop('y', axis=1)
    y = df['y']

    negatives = df[df['y'] == 0].drop('y', axis=1).values
    positives = df[df['y'] == 1].drop('y', axis=1).values

    plt.scatter(negatives[:, 0], negatives[:, 1], c='blue')
    plt.scatter(positives[:, 0], positives[:, 1], c='red')
    plt.show()

def scatter_3d():
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=3, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                               n_classes=2, random_state=15, class_sep=1.0)
    df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1), columns=['f1', 'f2', 'f3', 'y'])
    df['y'] = df['y'].astype('int')
    X = df.drop('y', axis=1)
    y = df['y']

    negatives = df[df['y'] == 0].drop('y', axis=1).values
    positives = df[df['y'] == 1].drop('y', axis=1).values

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)

    ax.scatter(negatives[:, 0], negatives[:, 1], negatives[:, 2], c='blue')
    ax.scatter(positives[:, 0], positives[:, 1], positives[:, 2], c='red')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()

def set_substract():
    from ml.utils import bootstrap
    data = np.random.random_integers(0, 100, (5,2))
    m = len(data)
    resamples, idx = bootstrap(data)
    print(data)
    print(resamples)

    idx_sub = list(set(range(m)) - set(idx))
    leave_data = data[idx_sub, :]
    print(leave_data)

def adaboost_test():
    n_samples = 5000
    n_features = 10
    n_informative = 8
    random_state = 19
    n_clusters_per_class = 1
    max_depth = 3

    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=0, n_clusters_per_class=n_clusters_per_class,
                               n_classes=2, random_state=random_state, class_sep=1.0)
    columns = []
    for i in range(1, n_features+1):
        columns.append('f%d'%i)
    columns.append('y')
    df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1), columns=columns)
    df['y'] = df['y'].astype('int')
    X = df.drop('y', axis=1)
    y = df['y']

    # negatives = df[df['y'] == 0].drop('y', axis=1).values
    # positives = df[df['y'] == 1].drop('y', axis=1).values
    #
    # plt.scatter(negatives[:, 0], negatives[:, 1], c='blue')
    # plt.scatter(positives[:, 0], positives[:, 1], c='red')
    # plt.show()

    from sklearn.model_selection import train_test_split
    X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.3, random_state=random_state)

    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier(max_depth=max_depth)
    dtc.fit(X_train, y_train)
    print('DecisionTreeClassifier score: ', dtc.score(X_vali, y_vali))

    from ml.ensemble import AdaBoostClassifier
    abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth), n_estimators=50)
    abc.fit(X_train, y_train)
    print('AdaBoostClassifier score: ', abc.score(X_vali, y_vali))

    from sklearn.ensemble import AdaBoostClassifier as ada
    ad = ada(DecisionTreeClassifier(max_depth=max_depth), n_estimators=50)
    ad.fit(X_train, y_train)
    print('sklearn adaboost score: ', ad.score(X_vali, y_vali))

def tree_model():
    n_samples = 500
    n_features = 5
    n_informative = 5
    random_state = 20
    n_clusters_per_class = 1
    max_depth = 3

    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_informative, n_redundant=0,
                               n_clusters_per_class=n_clusters_per_class,
                               n_classes=2, random_state=random_state, class_sep=1.0)
    columns = []
    for i in range(1, n_features + 1):
        columns.append('f%d' % i)
    columns.append('y')
    df = pd.DataFrame(np.concatenate((X, y.reshape(-1, 1)), axis=1), columns=columns)
    df['y'] = df['y'].astype('int')
    X = df.drop('y', axis=1)
    y = df['y']

    # negatives = df[df['y'] == 0].drop('y', axis=1).values
    # positives = df[df['y'] == 1].drop('y', axis=1).values
    #
    # plt.scatter(negatives[:, 0], negatives[:, 1], c='blue')
    # plt.scatter(positives[:, 0], positives[:, 1], c='red')
    # plt.show()

    from sklearn.model_selection import train_test_split
    X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.3, random_state=random_state)

    models = []
    from ml.tree import DecisionTreeClassifier, DecisionTree
    from sklearn.tree import DecisionTreeClassifier as DTC
    from ml.ensemble import AdaBoostClassifier
    models.append(DecisionTreeClassifier())
    # models.append(DecisionTree())
    # models.append(DTC(max_depth=max_depth))
    models.append(AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=50))
    # models.append(AdaBoostClassifier(DTC(max_depth=max_depth), n_estimators=20))

    for model in models:
        model.fit(X_train, y_train)
        print('score: ', model.score(X_vali, y_vali))

def test_DecitionTreeRegressor():
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_boston
    dataset = load_boston()
    X, y, features = dataset['data'], dataset['target'], dataset['feature_names']
    X = pd.DataFrame(X, columns=features)
    y = pd.DataFrame(y, columns=['target'])
    data = pd.concat([X, y], axis=1)

    features = data.columns[:-1]
    target = data.columns[-1]

    from sklearn.model_selection import train_test_split
    X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=23)
    print('X_train shape: ', X_train.shape)
    print('X_vali shape: ', X_vali.shape)
    print('y_train shape: ', y_train.shape)
    print('y_vali shape: ', y_vali.shape)

    from ml.tree import DecisionTreeRegressor
    from sklearn.tree import DecisionTreeRegressor as DTR
    models = {}
    models['my_dtr'] = DecisionTreeRegressor(max_depth=5)
    models['sklearn_dtr'] = DTR(max_depth=5)

    for name, model in models.items():
        model.fit(X_train, y_train)
        print('%s score: %.8f'%(name, model.score(X_vali, y_vali)))


def test_gbdt():
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_boston
    dataset = load_boston()
    X, y, features = dataset['data'], dataset['target'], dataset['feature_names']
    X = pd.DataFrame(X, columns=features)
    y = pd.DataFrame(y, columns=['target'])
    data = pd.concat([X, y], axis=1)

    features = data.columns[:-1]
    target = data.columns[-1]

    from sklearn.model_selection import train_test_split
    X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size=0.2, random_state=25)
    print('X_train shape: ', X_train.shape)
    print('X_vali shape: ', X_vali.shape)
    print('y_train shape: ', y_train.shape)
    print('y_vali shape: ', y_vali.shape)

    from sklearn.tree import DecisionTreeRegressor as DTR
    dtr = DTR(max_depth=5)
    dtr.fit(X_train, y_train.values.reshape(-1))
    print('sklearn dtr score: ', dtr.score(X_vali, y_vali))

    from sklearn.ensemble import GradientBoostingRegressor as GBR
    import xgboost as xgb
    gbr = GBR(max_depth=5)
    gbr.fit(X_train, y_train)
    print('sklearn gbr score: ', gbr.score(X_vali, y_vali))

    from ml.tree import DecisionTreeRegressor
    mydtr = DecisionTreeRegressor(max_depth=5)
    mydtr.fit(X_train, y_train)
    print('my dtr score: ', mydtr.score(X_vali, y_vali))

    from ml.ensemble import GradientBoostingRegressor
    mygbr = GradientBoostingRegressor()
    mygbr.fit(X_train, y_train)
    print('my gbr score: ', mygbr.score(X_vali, y_vali))

    # dtrain = xgb.DMatrix(X_train, label=y_train[target])
    # params = {'max_depth': 10}
    # num_round = 10
    # r = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_round)
    # dtest = xgb.DMatrix(X_vali)
    # ypred = r.predict(dtest)
    # from sklearn.metrics import r2_score
    # print('xgboost score: ', r2_score(y_vali, ypred))

    # xgbr = xgb.XGBRegressor()
    # xgbr.fit(X_train, y_train)
    # print('xgboost score: ', xgbr.score(X_vali, y_vali))


if __name__ == '__main__':
    test_gbdt()
