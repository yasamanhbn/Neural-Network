import yaml
def read_configs():
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    momentum_val = config['momentum']
    num_epochs = config['num_epochs']
    input_type = config['input_type']
    cost_function = config['const_function']
    activation = config['activation']
    mean = config['mean']
    standard_deviation = config['std']
    bias_val = config['Bias']
    gamma = config['gamma']
    hidden_layers = config['hidden_layers']
    lr_schedule = config['lr_schedule']
    W_init = config['W_init']
# Access dataset parameters
    dataset_path = config['dataset']['path']
    num_classes = config['dataset']['num_classes']
    train_split = config['dataset']['train_split']
    


    return learning_rate, batch_size, momentum_val, num_epochs, input_type, cost_function, dataset_path, num_classes, train_split, activation, mean, standard_deviation, bias_val, hidden_layers, gamma, W_init, lr_schedule

def get_datasetPath():
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['dataset']['path']

def get_inputType():
    with open('config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config['input_type']
