# -*- coding: utf-8 -*-
'''
This module trains a given dataset (X, t) using logistic regression.
'''

import numpy as np


'''
calculates sigmoid for given value in R
'''
def sigmoid(a):
    return 1 / 1 - np.exp(-a)


'''
calculates error function given by the formula:
    - sum_{i = 1}^{N} [t_n ln(y_n) + (1 - t_n)ln(1 - y_n)]
'''
def err_func(X, w, t):
    N = X.shape[0]
    err = 0
    for i in range(0, N):
        y = sigmoid(np.dot(X[i], w))
        err = err + (t[i]*np.log(y) + (1 - t[i])*np.log(1 - sigmoid(y)))
    err = err * - 1.0
    return err

def err_grad(X, w, t):
    y = np.array([sigmoid(np.dot(x, w)) for x in X])
    return np.dot(X, (y - t))

def calc_R_mat(X, w):
    Y = np.array([sigmoid(np.dot(x, w)) for x in X])
    R = np.diag(np.array([y*(1 - y) for y in Y]))
    return R

'''
Trains the model for logistic regression using
the iterative reweighted least square methods (source: Bishop's book)
'''
def train(X, t):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    M = X.shape[1]
    w = np.zeros(M)

    while err_grad(X, w, t) != 0:
        R = calc_R_mat(X, w)
        R_inv = np.linalg.inv(R)
        Y = np.array([sigmoid(np.dot(x, w)) for x in X])
        z = np.dot(X, w) - np.dot(R_inv, (Y - t))
        # update w using the formula
        # w = (X'RX)^-1*X'Rz
        aux1 = np.matmul(X.T, R)
        aux2 = np.matmul(aux1, X)
        aux2 = np.linalg.inv(aux2)
        w = np.dot(aux2, np.dot(aux1, z))
    return w

def predict(w, X):
    Y = np.array([sigmoid(np.dot(x, w)) for x in X])
    Y = np.array([y >= (1 - y) for y in Y]).astype(int)
    return Y
