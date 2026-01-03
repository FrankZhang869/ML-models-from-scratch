import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv("/kaggle/input/iris-classification/IRIS.csv")
for i in range(len(df)):
    if (df.at[i, "species"] == "Iris-setosa"):
        df.at[i, "species"] = 0
    elif (df.at[i, "species"] == "Iris-virginica"):
        df.at[i, "species"] = 1
    else:
        df.at[i, "species"] = 2

train, test = train_test_split(df, test_size=0.2)
X_train, Y_train, X_test, Y_test = train[train.columns[:4]], train[train.columns[4]], test[test.columns[:4]], test[test.columns[4]]

def scale_data (X):
    features= [X[col].to_numpy() for col in X.columns]
    for i in range(len(features)):
        features[i] = (features[i]-np.mean(features[i]))/np.std(features[i])
    scaled_data = np.stack(features, axis = 1)
    return scaled_data.T

X_train = scale_data(X_train) # shape is (4, 120)
X_test = scale_data(X_test) # shape is (4, 30)

def init_vars (X):
    m, n = X.shape # m is 4 and n is 120
    theta = np.zeros((3,m)) # 3, 4
    bias = np.zeros((3, 1))
    return theta, bias
def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))
def one_hot_encode (Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
def get_predictions(y_pred):
    return np.argmax(y_pred, 0)
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size
def derivative_loss(theta, bias, X, Y):
    m, n = X.shape
    Z = theta.dot(X) + bias # (3, 4).dot(4, 120) - (3, 120)
    y_pred = softmax(Z)
    return 1/n * (y_pred - one_hot_encode(Y)).dot(X.T), 1/n * np.sum(y_pred - one_hot_encode(Y), axis=1, keepdims=True)
def gradient_descent (theta, bias, X, Y, num_iters, lr):
    for i in range(num_iters):
        theta_loss, bias_loss = derivative_loss(theta, bias, X, Y)
        theta -= lr * theta_loss
        bias -= lr * bias_loss
        Z = theta.dot(X) + bias # (3, 4).dot(4, 120) - (3, 120)
        y_pred = softmax(Z)
        if (i % 100 == 0):
            print("iterations: ", i)
            print("accuracy: ", get_accuracy(get_predictions(y_pred), Y))

theta, bias = init_vars(X_train)
Y_train = np.array(Y_train)
Y_train = Y_train.astype(int)
gradient_descent(theta, bias, X_train, Y_train, 2000, 0.1)
    
        

