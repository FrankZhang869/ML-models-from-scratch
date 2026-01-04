import numpy as np

def init_vars (X):
  m, n = X.shape
  theta = np.zeros((1, m)) # (1, 30)
  bias = np.zeros((1, 1)) # (1, 1)
  return theta, bias

def sigmoid (Z):
  Z = np.clip(Z, -500, 500)
  return 1 / (1 + np.exp(-Z))

def deriv_loss (theta, bias, X, Y):
  Z = theta.dot(X) + bias 
  y_pred = sigmoid(Z)
  theta_loss = (y_pred - Y).dot(X.T) # (1, 30)
  bias_loss = np.sum(y_pred - Y, keepdims=True) # (1, 1)
  return theta_loss, bias_loss

def get_predictions (y_pred):
  return (y_pred >= 0.5).astype(int)

def get_accuracy (predictions, Y):
  return np.sum(predictions == Y) / Y.size

def gradient_descent (theta, bias, X, Y, num_iters, lr):
  for i in range(num_iters):
    theta_loss, bias_loss = deriv_loss(theta, bias, X, Y)
    theta -= theta_loss * lr
    bias -= bias_loss * lr
    Z = theta.dot(X) + bias 
    y_pred = sigmoid(Z)
    if (i % 1000 == 0):
      print("iterations: ", i)
      print("accuracy: ", get_accuracy(get_predictions(y_pred), Y))
