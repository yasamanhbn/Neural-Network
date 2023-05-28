
from matplotlib import pyplot as plt
import numpy as np

""" 
Reference: https://www.kaggle.com/code/vassiliskrikonis/cifar-10-analysis-with-a-neural-network

"""
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def loadCifar_train(datasetPath):
    print("start reading dataset...")
    train_batch_1 = unpickle(datasetPath + '/data_batch_1')
    train_batch_2 = unpickle(datasetPath +'/data_batch_2')
    train_batch_3 = unpickle(datasetPath + '/data_batch_3')
    train_batch_4 = unpickle(datasetPath + '/data_batch_4')
    train_batch_5 = unpickle(datasetPath + '/data_batch_5')
    meta_data = unpickle(datasetPath + '/batches.meta')

    X_train_origin = np.concatenate([train_batch_1[b'data'], train_batch_2[b'data'], train_batch_3[b'data'], train_batch_4[b'data'], train_batch_5[b'data']])

    """Load meta file"""
    label_names = meta_data[b'label_names']
    y_train_origin = np.concatenate([train_batch_1[b'labels'], train_batch_2[b'labels'], train_batch_3[b'labels'], train_batch_4[b'labels'], train_batch_5[b'labels']])

    idx = np.arange(X_train_origin.shape[0])
    np.random.shuffle(idx)
    X_train_origin = X_train_origin[idx]
    y_train_origin = y_train_origin[idx]
    print("reading dataset is done")
    return X_train_origin, y_train_origin, label_names


def loadCifar_test(datasetPath):
    print("start reading dataset...")
    test_batch = unpickle(datasetPath + '/test_batch')
    meta_data = unpickle(datasetPath + '/batches.meta')

    X_test_origin = test_batch[b'data']

    """Load meta file"""
    label_names = meta_data[b'label_names']
    y_test = test_batch[b'labels']

    print("reading dataset is done")
    return X_test_origin, y_test, label_names