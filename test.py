import numpy as np
import utils
from sklearn.model_selection import train_test_split
import dataloaders
import datasets
import deeplearning


def testProcess(model):
    # Load config data
    _, _, _, _, input_type, _ , dataset_path, _ , _ , _ , _ , _ , _ , _, gamma, _, _ = utils.read_configs()
    X_test_origin, y_test, label_names = dataloaders.loadCifar_test(dataset_path)
    X_test = datasets.test_preprocessing(input_type, X_test_origin)
    deeplearning.evaluation(model, X_test, y_test, label_names, gamma)

def regression_test(model, X_test, y_test):
    deeplearning.regression_evaluation(model, X_test, y_test)

if __name__ == "__main__":
    testProcess([])

    
