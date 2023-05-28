import numpy as np 

def regression_loss(pred,actuals , W, gamma):
  pred = np.squeeze(pred)
  leg = (gamma / (2 * len(pred))) * np.sum(np.square(W)) #legularization
  return (np.sum(1/2 * (pred-actuals)**2) / pred.shape[0]) + leg


def regression_loss_grad(pred, actuals):
  pred = np.squeeze(pred)
  result = (pred - actuals) / pred.shape[0]
  return   result.reshape(pred.shape[0], 1)