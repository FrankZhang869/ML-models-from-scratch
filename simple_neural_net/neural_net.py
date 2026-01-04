def init_vars (X):
    m, n = X.shape # m is 561, n is 7352
    W1 = np.random.rand(128, m) - 0.5 # (128, 561)
    b1 = np.random.rand(128, 1) - 0.5 # (128, 1)
    W2 = np.random.rand(32, 128) - 0.5
    b2 = np.random.rand(32, 1) - 0.5
    W3 = np.random.rand(6, 32) - 0.5
    b3 = np.random.rand(6, 1) - 0.5
    return W1, b1, W2, b2, W3, b3
def relu (Z):
    return np.maximum(0, Z)
def deriv_relu (Z):
    return Z > 0
def softmax (Z):
    return np.exp(Z) / sum(np.exp(Z))
def forward_prop (W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1 # (128, 7352)
    A1 = relu(Z1) # (128, 7352)
    Z2 = W2.dot(A1) + b2 # (32, 7352)
    A2 = relu(Z2) # (32, 7352)
    Z3 = W3.dot(A2) + b3 # (6, 7352)
    A3 = softmax(Z3) # (6, 7352)
    return Z1, A1, Z2, A2, Z3, A3
def one_hot_encode (Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
def back_prop (Z1, A1, Z2, A2, Z3, A3, Y, X):
    m = Y.size
    one_hot_Y = one_hot_encode(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1/m * dZ3.dot(A2.T)
    db3 = 1/m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * deriv_relu(Z2) # (32, 7352)
    dW2 = 1/m * dZ2.dot(A1.T) # (32, 128)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_relu(Z1) # (128, 7352)
    dW1 = 1/m * dZ1.dot(X.T) # (128, 561)
    db1 = 1/m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3
def update_vars (W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, lr):
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3
    return W1, b1, W2, b2, W3, b3
def get_predictions (A3):
    return np.argmax(A3, 0)
def get_accuracy (predictions, Y):
    return np.sum(predictions == Y) / (Y.size)
def gradient_descent (W1, b1, W2, b2, W3, b3, X, Y, num_iters, lr):
    for i in range(num_iters):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, Y, X)
        W1, b1, W2, b2, W3, b3 = update_vars (W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, lr)
        if (i % 100 == 0):
            print("iterations: ", i)
            print("accuracy: ", get_accuracy(get_predictions(A3), Y))
    return W1, b1, W2, b2, W3, b3
def test_predictions (W1, b1, W2, b2, W3, b3, X, Y):
    Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    return get_accuracy(get_predictions(A3), Y)
