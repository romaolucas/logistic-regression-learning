import numpy as np

class MultLogRegClassifier:

    def __init__(self, X, T, thres=10**-4):
        self.X = X
        self.T = T
        self.thres = thres
        n, k = T.shape
        m = X.shape[1]
        self.K = k
        self.M = m
        self.N = n
        self.W = np.zeros((m + 1, k))


    '''
    computes the activation vector for a given x
    '''
    def activation_vec(self, x):
        return np.dot(self.W, x)

    '''
    calculates the softmax function for x at
    a given index k, k in 1..K
    '''
    def softmax(self, activations, k):
        return np.exp(activations[k - 1]) / np.sum(np.apply_along_axis(np.exp, 0, activations))

    def calculate_y(self, X):
        activations = np.array([self.activation_vec(x) for x in X])
        Y = []
        N = X.shape[0]
        for n, x in zip(range(0, N), X):
            '''
            Ynk = softmax(activations[n], k)
                = exp(activations[n][k -1] /
                sum(exp(activations[n][j])
                j = 0...K-1
            '''
            Y.append([self.softmax(activations[n], k) for k in range(1, self.K + 1)]) 
        return Y

    def err_func(self):
        err = 0
        Y = self.calculate_y(self.X)
        for n in range(0, self.N):
            for k in range(0, self.K):
                err += self.T[n][k]*np.log(Y[n][k])
        return -err

    def err_grad(self):
        Y = self.calculate_y(self.X)
        return np.dot(self.X.T, (Y - T))

    def hessian(self):
        HT = np.zeros(self.M, self.K, self.M, self.K)
        for i in range(0, self.K):
            for j in range(0, self.K):
                
            
        



