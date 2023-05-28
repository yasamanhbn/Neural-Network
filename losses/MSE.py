import numpy as np 

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def one_hot(actuals):
  y_onehot = np.zeros((actuals.size, 10))
  y_onehot[np.arange(actuals.size), actuals] = 1 #make actual label onehot
  return y_onehot


def MSE(pred,actuals , W, gamma):
  pred = sigmoid(pred)
  y_onehot = one_hot(actuals)
  leg = (gamma / (2 * len(pred))) * np.sum(np.square(W)) #regularization
  return (np.sum(1/2 * (pred-y_onehot)**2) / pred.shape[0]) + leg


def MSE_grad(pred, actuals):
  predicted = sigmoid(pred)
  y_onehot = one_hot(actuals)
  return ((predicted - y_onehot) / pred.shape[0]) * predicted * (1 - predicted)