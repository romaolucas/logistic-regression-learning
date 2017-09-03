import numpy as np

class MultLogRegClassifier:

    def __init__(self, X, T, thres=10**-4):
        self.X = X
        self.T = T
        self.thres = thres
        n, k = T.shape
        m = X.shape[1] + 1
        self.K = k
        self.M = m
        self.N = n
        self.W = np.zeros((m, k))


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
        return np.dot(self.X.T, (Y - self.T))

    def hessian(self):
        HT = np.zeros(self.M, self.K, self.M, self.K)
        Y = self.calculate_y(self.X)
        for i in range(0, self.K):
            for j in range(0, self.K):
                r = np.multiply(Y[:, i], (int(i == j) - Y[:, j]))
                HT[:, i, :, j] = np.dot(self.X.T * r, self.X)
        return HT.reshape(self.M*self.K, self.M*self.K, order='F')
        
    def train(self):
        self.X = np.hstack((np.ones((self.X.shape[0], 1)), self.X))
        iterations = 0
        max_iterations = 200 
        converged = False
        err = self.err_func()
        while not converged:
            grad = self.err_grad()
            H = self.hessian()
            inv_H = np.linalg.inv(H)
            aux_W = self.W.flatten(order='F')
            aux_W = aux_W - np.dot(inv_H, grad.flatten(order='F'))
            self.W = aux_W.reshape(self.M, self.K, order='F')
            updated_err = self.err_func()
            iterations += 1
            converged = (err - updated_err) < self.thres or iterations > max_iterations
        print("Convergiu apos {} iteracoes".format(iterations))

