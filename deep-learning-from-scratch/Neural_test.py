import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1+np.exp(-x))

def identity_function(x):
    return x


# 入力層(1)
X = np.array([1.0,0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)


# 隠れ層(2)
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2)
Z2 = sigmoid(A2)


# 出力層(3)
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3)
Y = identity_function(A3)


# print(A1)
# print(Z1)
print(Y)



"""
X = np.array([1,2])
W = np.array([[1,3,5], [2,4,6]])

Y = np.dot(X,W)
print(Y)
"""