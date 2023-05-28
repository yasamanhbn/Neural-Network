import numpy as np
def train_preprocessing(input_type, X_train):
    print("start preprocessing...")
    X_train = X_train.astype('float64')
    X_train_res = X_train
    if input_type == 'Normalized':
        X_train_res = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
    elif input_type == 'Standardized':
        train_mean = X_train.mean(axis=0) #train mean 
        X_train_std = X_train.std(axis=0) #train std 
        X_train_res = (X_train - train_mean) / X_train_std #Standardized train
    print("preprocessing is done")
    return X_train_res

def test_preprocessing(input_type, X_test):
    X_test = X_test.astype('float64')
    X_test_res = X_test
    if input_type == 'Normalized':
        X_test_res = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))
    elif input_type == 'Standardized':
        test_mean = X_test.mean(axis=0) #test mean 
        X_test_std = X_test.std(axis=0) #test std
        X_test_res = (X_test - test_mean) / X_test_std #Standardized test
    return X_test_res