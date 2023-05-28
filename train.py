import numpy as np
import utils
from sklearn.model_selection import train_test_split
import dataloaders
import datasets
import deeplearning

def trainProcess():
    # Load config data
    learning_rate, batch_size, momentum_val, num_epochs, input_type, cost_function, dataset_path, num_classes, train_split, activation, meann, standard_deviation, bias_val, hidden_layers, gamma,W_init, lr_schedule = utils.read_configs()
    X_train_origin, y_train_origin, _ = dataloaders.loadCifar_train(dataset_path)
    X_train_pre = datasets.train_preprocessing(input_type, X_train_origin)
    X_train, X_val, y_train, y_val = train_test_split(X_train_pre, y_train_origin, test_size=train_split, shuffle=True, random_state=42)
    hidden_layers.insert(0, X_train.shape[1])
    hidden_layers.append(num_classes)
    model = deeplearning.get_model(hidden_layers, meann, standard_deviation, bias_val, learning_rate, momentum_val, activation, gamma, W_init)
    train_acc, train_loss, test_acc, test_loss, model = deeplearning.fit(model, X_train, y_train, X_val, y_val, cost_function, epochs=num_epochs, batchsize=batch_size, gamma=gamma, lr_schedule=lr_schedule)
    utils.plot_acc_loss(train_acc, train_loss, test_acc, test_loss)
    return model

def regression_train(X_train, X_val, y_train, y_val):
    learning_rate, batch_size, momentum_val, num_epochs, _, cost_function, _, _, _, activation, meann, standard_deviation, bias_val, hidden_layers, gamma, W_init, lr_schedule = utils.read_configs()
    hidden_layers.insert(0, X_train.shape[1])
    hidden_layers.append(1)
    model = deeplearning.get_model(hidden_layers, meann, standard_deviation, bias_val, learning_rate, momentum_val, activation, gamma, W_init)
    _, train_loss, _, test_loss, model = deeplearning.fit(model, X_train, y_train, X_val, y_val, cost_function, epochs=num_epochs, batchsize=batch_size, gamma=gamma, lr_schedule=lr_schedule)
    utils.plot_loss(train_loss, test_loss)
    return model


if __name__ == "__main__":
    trainProcess()


