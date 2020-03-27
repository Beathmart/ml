import numpy as np
from matplotlib import pyplot as plt
from ml.metrics import *

class LogisticRegression:

    def __init__(self, lr=0.001, threshold=0.5, regularization='L2', penalty_rate=0.1):
        self.lr = lr
        self.theta = None
        self.bias = None
        self.loss = []
        self.predict_threshold = threshold
        self.regularization = regularization
        self.penalty_rate = penalty_rate
        self.n_classes = None

    def _sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def _softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=1).reshape(-1, 1)

    def _loss(self, h, y):
        m = y.shape[0]
        return -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + self.penalty_rate / 2 / m * np.sum(self.theta)
        # return -1 / m * (np.dot(y.T, np.log1p(h)) + np.dot((1 - y).T, np.log1p(1 - h))) + self.penalty_rate / 2 / m * sum(self.theta)

    def _multinomial_loss(self, h, y, p=1e-10):
        m = y.shape[0]
        # y, h = get_one_hot(y, c), get_one_hot(h, c)
        p = p * np.ones((1, self.n_classes))
        return -1 / m * np.sum(y * np.log(h + p))


    def _get_one_hot(self, n_classes, targets):
        return np.eye(n_classes)[targets]

    def _optimizer_binary(self, X, y, iter=1000):
        # X: mxn
        # y: mx1
        m, n = X.shape
        y = np.array([y]).T

        # initialization
        self.theta = np.random.randn(n, 1)  # nx1
        self.bias = 0

        for i in range(iter):
            z = np.dot(X, self.theta) + self.bias
            h = self._sigmoid(z)
            self.theta = self.theta - self.lr / m * (np.dot(X.T, (h - y)) + self.penalty_rate * self.theta)
            self.bias = self.bias - self.lr / m * np.sum(h - y)
            self.loss.append(self._loss(self._sigmoid(np.dot(X, self.theta) + self.bias), y))
        print('training completed')
        # print('theta: ', self.theta, '\nbias: ', self.bias)

    def _optimizer_category(self, X, y, iter=1000):
        # X: mxn
        # y: mxc, c>2
        m, n = X.shape
        # self.n_classes = len(np.unique(y))
        # y = np.array([y]).T
        y = self._get_one_hot(self.n_classes, y) # mxc

        # initialization
        self.theta = np.random.randn(n, self.n_classes)  # nxc
        self.bias = np.zeros((1, self.n_classes))  # 1xc

        for i in range(iter):
            z = np.dot(X, self.theta) + self.bias
            h = self._sigmoid(z)
            self.theta = self.theta - self.lr / m * (np.dot(X.T, (h - y)) + self.penalty_rate * self.theta)
            self.bias = self.bias - self.lr / m * np.sum((h - y), axis=0)
            self.loss.append(self._loss(self._sigmoid(np.dot(X, self.theta) + self.bias), y))
        print('training completed')
        # print('theta: ', self.theta, '\nbias: ', self.bias)

    def _optimizer_softmax(self, X, y, iter=1000):
        # X: mxn
        # y: mxc, c>2
        m, n = X.shape
        # self.n_classes = len(np.unique(y))
        # y = np.array([y]).T
        y = self._get_one_hot(self.n_classes, y) # mxc

        # initialization
        self.theta = np.random.randn(n, self.n_classes)  # nxc
        self.bias = np.random.randn(1, self.n_classes)  # 1xc

        for i in range(iter):
            z = np.dot(X, self.theta) + self.bias
            h = self._softmax(z)
            self.theta = self.theta - self.lr / m * (np.dot(X.T, (h - y)) + self.penalty_rate * self.theta)
            self.bias = self.bias - self.lr / m * np.sum((h - y), axis=0)
            self.loss.append(self._multinomial_loss(self._softmax(np.dot(X, self.theta) + self.bias), y))
        print('training completed, iteration ', iter)
        # print('theta: ', self.theta, '\nbias: ', self.bias)

    def fit(self, X, y):
        # X: mxn
        # y: mx1
        self.n_classes = len(np.unique(y))
        if self.n_classes == 2:
            self._optimizer_binary(X, y)
        elif self.n_classes > 2:
            self._optimizer_softmax(X, y)
        else:
            raise ValueError('class number must be bigger than 1')


    def predict_proba(self, X):
        z = np.dot(X, self.theta) + self.bias
        if self.n_classes == 2:
            pred = self._sigmoid(z).reshape(-1)
        else:
            pred = self._softmax(z)
        return pred

    def predict(self, X):
        ## output shape (m,)
        if self.n_classes == 2:
            return (self.predict_proba(X) + (1 - self.predict_threshold)).astype('int').reshape(-1)
        elif self.n_classes > 2:
            return np.argmax(self.predict_proba(X), axis=1).reshape(-1)

    def evaluation(self, y_hat, y):
        error_rate = 0
        y_hat = y_hat.reshape(y.shape)
        for i in range(len(y)):
            if y_hat[i] != y[i]:
                error_rate += 1
        return 1 - error_rate / len(y)

    def score(self, X, y):
        return self.evaluation(self.predict(X), y)

    def history(self):
        return np.array([range(len(self.loss)), self.loss])

if __name__ == '__main__':
    # m = 1000
    # n = 2
    # X = np.random.randn(m, n)
    #
    # delta = 1.75
    # X[:m//2] += np.array([delta, delta])
    # X[m//2:] += np.array([-delta, -delta])
    #
    # y = np.array([0] * (m//2) + [1] * (m//2))
    #
    # # plt.scatter(X[:, 0], X[:, 1])
    # plt.scatter(X[:m//2, 0], X[:m//2, 1], color='red')
    # plt.scatter(X[m//2:, 0], X[m//2:, 1], color='blue')
    # plt.show()
    #
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=19)
    # print(
    #       'X_train shape: ',X_train.shape,
    #       'X_test shape: ', X_test.shape,
    #       'y_train shape: ', y_train.shape,
    #       'y_test: ', y_test.shape
    #       )
    # model = LogisticRegression(lr=0.01)
    # model.fit(X_train, y_train, iter=1000)
    # history = model.history()
    # plt.plot(history[0], history[1])
    # plt.show()
    # y_hat = model.predict(X_train)
    # print('score on train set: ', model.evaluation(y_hat, y_train))
    # print('score on test set: ', model.score(X_test, y_test))

    #### ----********************-----------------
    # sklearn model / test for pr curve
    # from sklearn.linear_model import LogisticRegression as lr
    # from sklearn.model_selection import train_test_split
    # from sklearn.datasets import make_classification
    # from sklearn.metrics import log_loss
    # from sklearn.metrics import precision_recall_curve as prc
    # from sklearn.metrics import roc_curve as ror_curve_sklearn
    # from sklearn.metrics import auc as auc_sklearn
    # from sklearn.neighbors import KNeighborsClassifier
    #
    # X, y = make_classification(n_samples=5000, n_features=10, n_informative=10, n_redundant=0, n_classes=2)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    # print(
    #       'X_train shape: ',X_train.shape,
    #       'X_test shape: ', X_test.shape,
    #       'y_train shape: ', y_train.shape,
    #       'y_test: ', y_test.shape
    #       )
    #
    # models = []
    # models.append(('my_lr', LogisticRegression(lr=0.01)))
    # # models.append(('LogisticRegression', lr()))
    # # models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
    #
    # for clf_name, clf in models:
    #     clf.fit(X_train, y_train)
    #     pred_train = clf.predict(X_train) # (m,)
    #     pred_test = clf.predict(X_test) # (m,)
    #     print(clf_name)
    #     print('train set acc: ', clf.score(X_train, y_train))
    #     print('test set acc: ', clf.score(X_test, y_test))
    #     # print('log loss on train set: ', category_log_loss(pred_train, y_train))
    #     print('confusion matrix:\n', '(tp, fp, fn, tn)\n', confusion_matrix(y_train, pred_train))
    #     print('\n')
    #
    #     proba_train = clf.predict_proba(X_train)
    #
    #     precisions, recalls, thresholds = prc(y_train, proba_train)
    #     # print(sorted(proba_train)[-1:-11:-1])
    #     # print(thresholds[-1:-11:-1])
    #     plt.plot(recalls, precisions, 'blue', label='sklearn function')
    #     print('sklearn auc: ', auc_sklearn(recalls, precisions))
    #
    #     precisions_, recalls_ = precision_recall_curve(y_train, proba_train)
    #     plt.plot(recalls_, precisions_, 'red', linestyle='--', label='my function')
    #     print('my acu: ', auc(recalls_, precisions_))
    #
    #     print(len(precisions), len(precisions_))
    #     plt.xlabel('recall')
    #     plt.ylabel('precision')
    #     plt.title('precision_recall_curve')
    #     plt.legend()
    #     # plt.xlim((-0.1, 1.1))
    #     # plt.ylim(((-0.1, 1.1)))
    # plt.show()

    ### ----********************-----------------
    ## test for roc curve
    from sklearn.linear_model import LogisticRegression as lr
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    from sklearn.metrics import log_loss
    from sklearn.metrics import precision_recall_curve as prc
    from sklearn.metrics import roc_curve as ror_curve_sklearn
    from sklearn.metrics import auc as auc_sklearn


    X, y = make_classification(n_samples=5000, n_features=10, n_informative=10, n_redundant=0, n_classes=2, n_clusters_per_class=1, random_state=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    print(
          'X_train shape: ',X_train.shape,
          'X_test shape: ', X_test.shape,
          'y_train shape: ', y_train.shape,
          'y_test: ', y_test.shape
          )

    models = []
    models.append(('my_lr', LogisticRegression(lr=0.01)))
    models.append(('LogisticRegression', lr()))
    # models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))

    for clf_name, clf in models:
        clf.fit(X_train, y_train)
        pred_train = clf.predict(X_train) # (m,)
        pred_test = clf.predict(X_test) # (m,)
        print(clf_name)
        print('train set acc: ', clf.score(X_train, y_train))
        print('test set acc: ', clf.score(X_test, y_test))
        # print('log loss on train set: ', category_log_loss(pred_train, y_train))
        print('confusion matrix:\n', '(tp, fp, fn, tn)\n', confusion_matrix(y_train, pred_train))
        print('\n')

        # proba_train = clf.predict_proba(X_train)
        #
        # fprs, tprs, thresholds = ror_curve_sklearn(y_train, proba_train)
        # plt.plot(fprs, tprs, 'blue', marker='x', label='sklearn function')
        # print('sklearn auc: ', auc_sklearn(fprs, tprs))
        #
        # tprs_, fprs_ = roc_curve(y_train, proba_train)
        # plt.plot(fprs_, tprs_, 'red', linestyle='--', label='my function')
        # print('my acu: ', auc(fprs_, tprs_))
        #
        # print(len(tprs), len(tprs_))
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('roc_curve')
        # plt.legend()
        # plt.xlim((-0.1, 1.1))
        # plt.ylim(((-0.1, 1.1)))
    # plt.show()

    #### ------***************************----------
    # from sklearn.datasets import make_classification
    # X, y = make_classification(n_samples=5000, n_features=10, n_informative=10, n_redundant=0, n_classes=3)
    #
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
    # print(
    #       'X_train shape: ',X_train.shape,
    #       'X_test shape: ', X_test.shape,
    #       'y_train shape: ', y_train.shape,
    #       'y_test: ', y_test.shape
    #       )
    #
    # model = LogisticRegression(lr=0.01)
    # model.fit(X_train, y_train, iter=1000)
    #
    # history = model.history()
    # plt.plot(history[0], history[1])
    # plt.show()
    #
    # print('score on train set: ', model.score(X_train, y_train))
    # print('score on test set: ', model.score(X_test, y_test))