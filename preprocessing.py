import numpy as np

def get_one_hot(labels, n):
    return np.eye(n)[labels]