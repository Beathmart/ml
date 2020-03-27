import time
import os
from functools import wraps
import pandas as pd
import numpy as np

def print_run_time(func):
    '''
    decorator for timer
    :param func: a function
    :return:
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        '''
        decorator function
        :param args:
        :param kwargs:
        :return:
        '''
        start = time.time()
        func(*args, **kwargs)
        print('current function [%s] run time is  %.2fs'%(func.__name__, time.time()-start))
    return wrapper


def load_csv(path):
    if path.strip().split('.')[-1] == 'csv':
        df = pd.read_csv(path)
    else:
        raise Exception('this is not a csv file')
    return df

def bootstrap(data):
    m = len(data)
    idx = np.random.randint(m, size=m)
    resamples = data[idx, :]
    return resamples, idx

def set_substract(data, idx):
    m = len(data)
    idx_sub = list(set(range(m)) - set(idx))
    leave_data = data[idx_sub, :]
    return leave_data

def pd_to_np(x):
    if not isinstance(x, np.ndarray):
        x = x.values
    return x

def pd_to_np_1d(x):
    if not isinstance(x, np.ndarray):
        x = x.values
    return x.reshape(-1,)
