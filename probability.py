import numpy as np
import pandas as pd

def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def A(n, k):
    # Arrangement
    return factorial(n) // factorial(n - k)

def C(n, k):
    # Combination
    return A(n, k) // factorial(k)

def binomial(n, k, p):
    return C(n, k) * (p ** k) * ((1 - p) ** (n - k))

def joint_probability(X, p, func):
    # X: sets for n and k
    # p: probability
    # func: distribution function
    L = 1
    for n, k in X:
        L *= func(n, k, p)
    return L

def MLE(X, probability_samples=100, dist=binomial):
    estimations = []
    for p in np.linspace(0, 1, probability_samples+1):
        L = joint_probability(X, p, dist)
        estimations.append(L)
    max_index = int(np.argmax(estimations))
    return max_index / probability_samples, estimations[max_index]

def normal_distribution(x, mean, std):
    return 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))

def information_entropy(x):
    x = prepare_data(x)

    m = len(x)
    classes = np.unique(x)
    n = len(classes)
    entropy = 0
    for i in range(n):
        num = len(x[x==classes[i]])
        p = num / m
        entropy += - p * np.log2(p)
    return entropy

def conditional_entropy(x, cond):
    x = prepare_data(x)
    cond = prepare_data(cond)

    arr = np.concatenate((x, cond), axis=1)
    m = len(x)
    classes = np.unique(arr[:, 0])
    n = len(classes)

    ce = 0
    for i in range(n):
        group = arr[arr[:, 0] == classes[i]]
        num = len(group)
        p = num / m
        entropy = information_entropy(group[:, 1])
        ce += p * entropy

    return ce

def information_gain(x, cond):
    return information_entropy(cond) - conditional_entropy(x, cond)

# def gini(x):
#     x = prepare_data(x)
#
#     g = 1
#     m = len(x)
#     c = np.unique(x)
#     n = len(c)
#     for i in range(n):
#         g -= (len(x[x==c[i]]) / m) ** 2
#     return g

def gini(d, weights=None):
    d = prepare_data(d)
    m = len(d)
    c = np.unique(d)
    n = len(c)

    g = 1
    for i in range(n):
        v = c[i]
        if weights is None:
            p = (len(d[d==v]) / m)
        else:
            p = np.sum(weights[d==v]) / np.sum(weights)
        g -= p ** 2
    return g


def gini_impurity(x, cond, weights=None):
    x = prepare_data(x)
    cond = prepare_data(cond)

    if weights is None:
        arr = np.concatenate((x, cond), axis=1)
    else:
        arr = np.concatenate((x, cond, weights), axis=1)

    m = len(x)
    classes = np.unique(arr[:, 0])
    n = len(classes)

    gini_impurity = 0
    for i in range(n):
        group = arr[arr[:, 0]==classes[i]]
        group_cond = group[:, 1]

        g = gini(group_cond, weights=weights)
        # m_cond = len(group_cond)
        # c_cond = np.unique(group_cond)
        # n_cond = len(c_cond)
        # g = 1
        # for j in range(n_cond):
        #     g -= (len(group_cond[group_cond==c_cond[j]]) / m_cond) ** 2

        if weights is None:
            num = len(group)
            p = num / m
            # print(p)
        else:
            p = np.sum(group[:, -1]) / np.sum(weights)

        gini_impurity += p * g
    return gini_impurity

def check_values(x):
    if type(x) == pd.core.frame.DataFrame or type(x) == pd.core.series.Series:
        x = x.values
    return x

def prepare_data(x):
    x = check_values(x)
    x = x.reshape(-1, 1)
    return x

if __name__ == '__main__':
    # x = np.array([0,0,0,0,1,1,1,0,1,1,1,0,1,0])
    # cond = np.array([0,0,1,1,1,0,1,0,1,1,1,1,1,0])
    # print(gini_impurity(x, cond))

    w = pd.DataFrame({'key': [0, 0, 1, 0, 1]})
    print(information_entropy(w))
