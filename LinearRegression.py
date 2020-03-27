import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001):
        self.lr = lr
        self.theta = None
        self.bias = None
        self.loss = []

    def fit(self, X, y, iter=100):
        # X: mxn
        # y: mx1
        m, n = X.shape
        self.theta = np.random.randn(n, 1) # nx1
        self.bias = 1
        for i in range(iter):
            h = self._hypothesis(X, self.theta, self.bias)  # mx1
            self.theta = self.theta - self.lr / m * np.dot(X.T, h - y)
            self.bias = self.bias - self.lr / m * np.sum(h - y)
            self.loss.append(self._loss(self._hypothesis(X, self.theta, self.bias), y))
            if (i+1) % 100 == 0:
                print('%dth iteration completed'%(i+1))
        print('training completed')
        print(self.theta, self.bias)

    def predict(self, X):
        pred = np.dot(X, self.theta) + self.bias
        print('pred shape in class: ', pred.shape)
        return pred

    def evaluation(self, y_hat, y):
        # y_hat, y: mx1

        print('the loss is %s'%(self._loss(y_hat, y)))
        # return mse

    def history(self):
        return np.array([range(len(self.loss)), self.loss])

    def _hypothesis(self, X, theta, bias):
        return np.dot(X, theta) + bias

    def _loss(self, h, y):
        return 1 / 2 / y.shape[0] * np.sum((h - y) ** 2)

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    X = np.random.randint(1, 10, (100, 2))
    theta = np.array([[2,3]]).T
    bias = 5
    y = np.dot(X, theta) + bias

    lr = LinearRegression(0.01)
    lr.fit(X, y, iter=1000)
    y_hat = lr.predict(X)
    lr.evaluation(y_hat, y)
    history = lr.history()
    print(history.shape)
    plt.plot(history[0], history[1])
    plt.show()