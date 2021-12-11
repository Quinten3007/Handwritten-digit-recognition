import numpy as np
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

X_train = np.array([i.flatten() for i in x_train]).T/255
X_test = np.array([i.flatten() for i in x_test]).T/255
np.random.seed(5)
m = X_train.shape[1]
print(y_test[0])


def init_params():
    W1, b1 = np.random.rand(64, 784) - 0.5, np.random.rand(64, 1) - 0.5
    W2, b2 = np.random.rand(64, 64) - 0.5, np.random.rand(64, 1) - 0.5
    W3, b3 = np.random.rand(10, 64) - 0.5, np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2, W3, b3


def ReLu(z):
    return np.maximum(z, 0)


def softmax(z):
    A = np.exp(z) / sum(np.exp(z))
    return A


def forward_prop(W1, b1, W2, b2, W3, b3, x):
    Z1 = W1.dot(x) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLu(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def ReLU_deriv(z):
    return z > 0


def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max()+1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, x, y):
    one_hot_y = one_hot(y)
    dZ3 = A3 - one_hot_y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(x.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1, b1 = W1-alpha*dW1, b1-alpha*db1
    W2, b2 = W2 - alpha * dW2, b2 - alpha * db2
    W3, b3 = W3 - alpha * dW3, b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3


def get_predictions(A3):
    return np.argmax(A3, 0)


def get_accuracy(predictions, y):
    print(predictions, y)
    return np.sum(predictions == y) / y.size


def gradient_descent(x, y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, x)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, x, y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 20 == 0:
            print('iteration: ', i)
            predictions = get_predictions(A3)
            print(get_accuracy(predictions, y))
    return W1, b1, W2, b2, W3, b3


W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, y_train, 0.10, 200)

Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X_test)
print(get_accuracy(get_predictions(A3), y_test))
print(np.argmax(forward_prop(W1, b1, W2, b2, W3, b3, X_test[:, 0].reshape(784, 1))[-1]))
