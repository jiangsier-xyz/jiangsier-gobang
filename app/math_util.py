#!/usr/bin/python
# -*- coding: utf-8 -*-


import random
import numpy as np


def max_sample(*args, key=None):
    max_arg = max(*args, key=key)
    results = []
    if key is None:
        key = lambda x: x
    for arg in list(*args):
        if key(arg) == key(max_arg):
            results.append(arg)
    if len(results) > 1:
        return random.sample(results, 1)[0]
    else:
        return max_arg


def softmax(x):
    priors = np.exp(x - np.max(x))  # 计算分子,利用softmax(x)=softmax(x+C)来保证不溢出
    priors = priors / np.sum(priors)
    return priors


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def prob(x):
    s = np.sum(x)
    if s < 1e-10:
        return prob(x + 1e-10)
    return x.astype(float) / s
