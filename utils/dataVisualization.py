import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

import numpy as np
import matplotlib.pyplot as plt
from dataloaders.cifarLoader import loadCifar_train
from utils.yamlRead import get_datasetPath

def dataset_visualization():
    dataset_path = get_datasetPath()
    X_train, labels, label_names = loadCifar_train(dataset_path)
    X_train = X_train.reshape(len(X_train), 3, 32, 32)
    X_train = X_train.transpose(0, 2, 3, 1) # Transpose  data
    print(X_train.shape)

    rows, columns = 10, 10
    fig = plt.figure(figsize=(18, 18))
    idx = 1
    for i in range(rows):
      indexes = np.where(labels==i)[0][:columns]
      for _, j in enumerate(indexes):
        fig.add_subplot(rows, columns, idx)
        plt.imshow(X_train[j])
        plt.axis('off')
        idx+=1
        plt.title(label_names[labels[indexes[0]]].decode('ascii'), fontsize = 10)
    plt.show()

# dataset_visualization()