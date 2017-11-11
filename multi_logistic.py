import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class MultLogRegClassifier:

    def __init__(self, X, t, labels, thres=10**-4):
        self.X = X
        self.T = np.zeros((X.shape[0], len(labels)))
        for n, lbl in enumerate(t):
            self.T[n][labels.index(lbl)] = 1
        self.thres = thres
        n, k = self.T.shape
        m = X.shape[1] + 1
        self.K = k
        self.M = m
        self.N = n
        self.labels = labels
        self.W = np.zeros((m, k))


    '''
    computes the activation vector for a given x
    '''
    def activation_vec(self, x):
        return np.dot(self.W.T, x)

    '''
    calculates the softmax function for x at
    a given index k, k in 1..K
    '''
    def softmax(self, activations, k):
        return np.exp(activations[k - 1]) / np.sum(np.apply_along_axis(np.exp, 0, activations))

    def calculate_y(self, X):
        Y = []
        N = X.shape[0]
        if N == 1:
            activations = self.activation_vec(X)
            return [self.softmax(activations, k) for k in range(1, self.K + 1)]
        activations = np.array([self.activation_vec(x) for x in X])
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
                err += self.T[n][k] * np.log(Y[n][k])
        return (-1) * err

    def err_grad(self):
        Y = self.calculate_y(self.X)
        return np.dot(self.X.T, (Y - self.T))

    def hessian(self):
        HT = np.zeros((self.M, self.K, self.M, self.K))
        Y = self.calculate_y(self.X)
        Y = np.array(Y)
        I = np.identity(self.K)
        for i in range(0, self.K):
            y_i = Y[:, i]
            for j in range(0, self.K):
                y_j = Y[:, j]
                r = y_i * (int(i == j) - y_j)
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
            print("err: {}, updated err: {}, err - updated_err: {}".format(err, updated_err, err - updated_err))
            converged = (err - updated_err) < self.thres or iterations > max_iterations
            err = updated_err
            iterations += 1
        print("Convergiu apos {} iteracoes".format(iterations))

    def predict(self, X):
        Y = []
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        if X.shape[0] == 1:
            y = self.calculate_y(X)
            max_idx = y.index(max(y))
            return self.labels[max_idx]
        Y = self.calculate_y(X)
        y_predicted = []
        for y in Y:
            max_idx = y.index(max(y))
            y_predicted.append(self.labels[max_idx])
        return y_predicted
    
    def calc_accuracy(self, X, t):
        y = self.predict(X)
        correctly_classified = 0
        for lbl_y, lbl_t in zip(y, t):
            if lbl_y == lbl_t:
                correctly_classified += 1
        return correctly_classified / len(t)

def main():
    X, t = load_iris(return_X_y=True)
    X_train, X_test, t_train, t_test = train_test_split(X, t)
    logReg = MultLogRegClassifier(X_train, t_train, list(set(t_train)))
    logReg.train()
    print("accuracy: {}".format(logReg.calc_accuracy(X_test, t_test)))

if __name__ == '__main__':
    main()
