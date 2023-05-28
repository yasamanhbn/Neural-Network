import numpy as np
import losses
import nets
import utils

# Compute activations of all network layers
def totalForward(model, xTrain):
    outputs = []
    input = xTrain
    model_counts = len(model)
    for i in range(model_counts):
      tmp = model[i].forward(input)
      outputs.append(tmp)
      input = tmp
    return outputs


#Reference: https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros
def iterate_minibatches(xTrain, yTrain, batchsize, shuffle):
    if shuffle:
        indices = np.random.permutation(len(xTrain))
    for start_idx in range(0, len(xTrain) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield xTrain[excerpt], yTrain[excerpt]

# Train network on a given batch of X and y
def train(nn,XTrain,yTrain, costFunction, gamma, epoch, epochs, lr_schedule):
    outputs = totalForward(nn,XTrain)
    pred = outputs[-1]

    # Compute the loss and the initial gradient
    if costFunction == 'Cross-Entropy':
      loss = losses.softmax_and_crossentropy(pred, yTrain, nn[len(nn) - 1].get_weigth(), gamma)
      loss_grad = losses.softmax_and_crossentropy_grad(pred, yTrain)
    elif costFunction == 'MSE':
      loss = losses.MSE(pred, yTrain, nn[len(nn) - 1].get_weigth(), gamma) 
      loss_grad = losses.MSE_grad(pred, yTrain)
    elif costFunction == 'Regression':
      loss = losses.regression_loss(pred, yTrain, nn[len(nn) - 1].get_weigth(), gamma) 
      loss_grad = losses.regression_loss_grad(pred, yTrain)

    grad_out = loss_grad
    initial_learning_rate = nn[len(nn) - 1].learning_rate
    decay = initial_learning_rate / epochs
    for i in range(1, len(nn)):
      grad_out = nn[len(nn) - i].backward(outputs[len(nn) - i - 1], grad_out)
      # Refereneces: https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
      if lr_schedule and isinstance(nn[len(nn) - i], nets.Layer):
        lrate = nn[len(nn) - i].learning_rate * 1 / (1 + decay * epoch)
        nn[len(nn) - i].learning_rate = lrate
    return loss

def predict(model,xTrain):
  result = totalForward(model,xTrain)[-1]
  return result.argmax(axis=-1)

def fit(model, X_train,y_train, X_val, y_val, costFunction, epochs, batchsize, gamma, lr_schedule):
  train_acc = []
  test_acc = []
  train_loss = []
  test_loss = []
  for epoch in range(epochs):
    eachLoss = 0
    for x, y in iterate_minibatches(X_train, y_train, batchsize=batchsize, shuffle=False):
      eachLoss+=train(model, x ,y, costFunction, gamma, epoch + 1, epochs, lr_schedule)
    
    train_acc.append(utils.get_accuracy(predict(model,X_train), y_train))
    test_acc.append(utils.get_accuracy(predict(model,X_val), y_val))
    train_loss.append(np.mean(eachLoss/((X_train.shape[0] / batchsize))))

    if costFunction == 'Cross-Entropy':
      test_loss.append(np.mean(losses.softmax_and_crossentropy(totalForward(model,X_val)[-1], y_val, model[len(model) - 1].get_weigth(), gamma)))

    elif costFunction == 'MSE':
      test_loss.append(np.mean(losses.MSE(totalForward(model,X_val)[-1], y_val, model[len(model) - 1].get_weigth(), gamma)))
    
    elif costFunction == 'Regression':
      test_loss.append(np.mean(losses.regression_loss(totalForward(model,X_val)[-1], y_val, model[len(model) - 1].get_weigth(), gamma)))

    if costFunction == 'Regression':
       print("Epoch:", epoch + 1, "  Train Loss:", round(train_loss[-1],4), "  Validation Loss:", round(test_loss[-1],4))

    else:
      print("Epoch:", epoch + 1, "   Train accuracy:", round(train_acc[-1], 4), "  Train Loss:", round(train_loss[-1],4), 
          "    Validation accuracy:", round(test_acc[-1],4), "  Validation Loss:", round(test_loss[-1],4))

  return train_acc, train_loss, test_acc, test_loss, model

def set_activation_func(activation):
  if activation =='ReLu':
    return nets.ReLu()
  elif activation =='LeakyReLu':
    return nets.LeakyReLu()
  elif activation == 'Tanh':
    return nets.Tanh()
  elif activation == 'Sigmoid':
    return nets.Sigmoid()

def get_model(hidden_layers, meann, standard_deviation, bias_val, learning_rate, momentum_val, activation, gamma, W_init):
    model = []
    for i in range(len(hidden_layers) - 1):
      model.append(nets.Layer(input_neuron=hidden_layers[i], output_neuron=hidden_layers[i+1], mean=meann, std=standard_deviation, bias_val=bias_val, learning_rate=learning_rate, momentum=momentum_val, gamma=gamma, W_init=W_init))
      if i != len(hidden_layers) - 2:
        model.append(set_activation_func(activation))
    return model

def evaluation(trained_model, X_test, y_test, label_names, gamma):
  y_hat = predict(trained_model, X_test)
  utils.get_evaluation_matrics(y_test, y_hat)
  print("Test Loss:")
  print(np.mean(losses.softmax_and_crossentropy(totalForward(trained_model,X_test)[-1], y_test, trained_model[len(trained_model) - 1].get_weigth(), gamma)))
  utils.make_confusion_matrix(y_test, y_hat, label_names)

def regression_evaluation(trained_model, X_test, y_test):
  print("Test Loss:")
  print(np.mean(losses.regression_loss(totalForward(trained_model,X_test)[-1], y_test, trained_model[len(trained_model) - 1].get_weigth(), 0)))  
