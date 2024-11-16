import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ds = pd.read_csv("insurance_data.csv")

x_train,y_train,x_test,y_test = train_test_split(ds[["age", "affordibility"]],ds.bought_insurance,test_size=0.2)

def sigmoid_numpy(y):
 return 1/(1+np.exp(-y))

def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = [max(i,epsilon) for i in y_pred]
    y_pred = [min(i,1-epsilon) for i in y_pred]# Clip values to avoid log(0) or log(1)
    return - (np.array(y_true) * np.log(y_pred) + (1 - np.array(y_true)) * np.log(1 - np.array(y_pred))).mean()

 
def gradient_descent(x1,x2,y_true,learning_rate,epochs):
  w1 = 1
  w2 = 1
  b = 0
  n = len(y_true)
  for i in range(epochs):
    y = x1 * w1 + x2 * w2 + b
    y_pred = sigmoid_numpy(y)
    loss = log_loss(y_true,y_pred)

    dw1 = np.dot(x1, y_pred - y_true) / n
    dw2 = np.dot(x2, y_pred - y_true) / n
    db = np.sum(y_pred - y_true) / n
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b -= learning_rate * db
    print(f"Epoch: {i+1}, Loss: {loss}, w1: {w1}, w2: {w2}")

  return w1,w2,b

gradient_descent(ds.age,ds.affordibility,ds.bought_insurance,0.01,1000000)
  
    
  
  