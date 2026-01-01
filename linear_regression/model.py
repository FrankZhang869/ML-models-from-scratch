import numpy as np
import pandas as pd

df = pd.read_csv("Ecommerce Customers")
df = df.drop("Email", axis=1)
df = df.drop("Address", axis=1)
df = df.drop("Avatar", axis=1)

def standardize(X):
    mean = np.mean(X, axis=0)      
    std = np.std(X, axis=0)       
    X_scaled = (X - mean) / std
    return X_scaled

df = standardize(df)
train, test = df[:400], df[400:]
x_train, y_train, x_test, y_test = train[train.columns[:-1]], train[train.columns[-1]], test[test.columns[:-1]], test[test.columns[-1]]

class LinearRegression1: # uses normal equation
  def fit (self, x, y):
    m, n = x.shape 
    self.theta = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y)) 
  def predict (self, x):
    y_predicted = np.dot(x, self.theta)
    return y_predicted

class LinearRegression2: # Uses gradient descent
  def __init__ (self, lr, num_iters):
    self.lr = lr
    self.num_iters = num_iters
  def fit (self, x, y):
    m, n = x.shape 
    self.theta = np.zeros(n)
    for _ in range(self.num_iters):
      y_pred = np.dot(x, self.theta)
      self.theta -= (self.lr * (np.dot(x.T, (y_pred - y))))  
  def predict (self, x):
    return np.dot(x, self.theta)
