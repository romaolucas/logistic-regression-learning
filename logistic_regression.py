# -*- coding: utf-8 -*-
'''
This module trains a given dataset (X, t) using logistic regression.
'''

import numpy as np


'''
calculates sigmoid for given value in R
'''
def sigmoid(a):
    return 1./( 1 + np.exp(-a))


'''
calculates error function given by the formula:
    - sum_{i = 1}^{N} [t_n ln(y_n) + (1 - t_n)ln(1 - y_n)]
'''
def err_func(X, w, t):
    N = X.shape[0]
    err = 0
    for i in range(0, N):
        y = sigmoid(np.dot(X[i].T, w))
        err = err + (t[i]*np.log(y) + (1 - t[i])*np.log(1 - y))
    err = err * - 1.0
    return err

def err_grad(X, w, t):
    y = np.array([sigmoid(np.dot(x, w)) for x in X])
    grad = np.dot(X.T, (y - t))
    return grad 

def calc_R_mat(X, w):
    Y = np.array([sigmoid(np.dot(x.T, w)) for x in X])
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
    iterations = 0
    converged = False
    err = err_func(X, w, t)
    thres = 10**-4
    while not converged:
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
        updated_err = err_func(X, w, t)
        print("err = {}, updated_err = {}, err - updated_err = {}".format(err, updated_err, err - updated_err))
        converged = (err - updated_err) < thres
        err = updated_err
        iterations += 1
    print("Convergiu apos {} iteracoes".format(iterations))
    return w

def predict(w, X):
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    Y = np.array([sigmoid(np.dot(x, w)) for x in X])
    Y = np.array([y >= (1 - y) for y in Y]).astype(int)
    return Y

def calc_accuracy(Y, T):
    correctly_classified = 0
    for y, t in zip(Y, T):
        if y == t:
            correctly_classified += 1
    return correctly_classified / Y.shape[0]

X = []
t = []
import csv
with open('clean2.data', mode='r') as csvfile:
    data_reader = csv.reader(csvfile, delimiter=',')
    for row in data_reader:
        X.append(row[0:166])
        t.append(row[-1])

X = np.array(X, dtype=float)
t = np.array(t, dtype=float)

from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.33)

w = train(X_train, t_train)
print("Prevendo dados de teste\n")
Y = predict(w, X_test)
print("Acuracia = ", calc_accuracy(Y, t_test))
