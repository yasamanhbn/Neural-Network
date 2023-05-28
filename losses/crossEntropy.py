import numpy as np 
def softmax(X):
    # exps = np.exp(X - np.max(X, axis=1, keepdims=True))
    exps = np.exp(X)
    return exps / np.sum(exps, axis=1, keepdims=True)

def softmax_and_crossentropy(pred,actuals, W, gamma):
    # Compute crossentropy
    pred =  pred - np.max(pred, axis=1, keepdims=True)
    # pred/=100
    m = len(pred)
    logits_for_answers = pred[np.arange(m), actuals]
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(pred), axis=1, keepdims=True))
    leg = (gamma / (2 * m)) * np.sum(np.square(W))  #regularization
    return xentropy + leg

def softmax_and_crossentropy_grad(pred,actuals):
    ones_for_answers = np.zeros_like(pred)
    ones_for_answers[np.arange(len(pred)),actuals] = 1
    softmax1 = softmax(pred)
    return (- ones_for_answers + softmax1) / pred.shape[0]