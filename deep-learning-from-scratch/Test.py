import numpy as np
import matplotlib.pylab as plt


A = np.array([[1,2,3],[4,3,2]])
B = np.array([[9,8],[7,6],[5,4]])
C = np.array([[2,9],[8,3]])

print(A)
print(B)
# print(np.ndim(A))
# print(A.shape)
print(np.dot(A, B))
print(np.dot(C, C))



"""
def step_function(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1+np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


x = np.arange(-50.0, 50.0, 0.1)
y = ReLU(x)#sigmoid(x) #step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 10)
plt.show()
"""