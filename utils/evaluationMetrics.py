from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import itertools
import numpy as np
import matplotlib.pyplot as plt

def make_confusion_matrix(y_true, y_pred, label_names): 
    newLabels = []
    for i in label_names:
        newLabels.append(str(i, "utf-8"))
    classes = newLabels
    figsize=(13, 13)
    text_size=6
    # Create the confustion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
    n_classes = cm.shape[0]

    # Plot the figure and make it pretty
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])
  
    # Label the axes
    ax.set(title="Confusion Matrix",
         xlabel="Prediced label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels,
         yticklabels=labels)
  
    # Make x-axis labels appear on bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Set the threshold for different colors
    threshold = (cm.max() + cm.min()) / 2.

    # Plot the text on each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)
    plt.show()


def get_accuracy(prediction, yTrue):
  assert(len(yTrue) == len(prediction))
  return np.mean(prediction==yTrue)


def get_evaluation_matrics(y_test, y_prediction):
    model_accuracy = get_accuracy(prediction=y_prediction, yTrue=y_test)
    print("Accuracy: {0:.3f}".format(model_accuracy * 100))
    
    recall_test = recall_score(y_test, y_prediction, average='micro')
    print("recall: {0:.3f}".format(recall_test * 100))
    
    precision_test = precision_score(y_test, y_prediction, average='micro')
    print("precision: {0:.3f}".format(precision_test * 100))
    
    f1_test = f1_score(y_test, y_prediction, average='micro')
    print("F1 score: {0:.3f}".format(f1_test * 100))


def plot_acc_loss(train_acc, train_loss, test_acc, test_loss):
  _, axes = plt.subplots(1, 2, figsize=(12, 3))
  plt.suptitle("Accuracy and Loss plot for train and validation data", size=14)
  axes[0].plot(train_acc, 'b',label='train accuracy')
  axes[0].plot(test_acc, 'r', label='test accuracy')
  axes[0].set_xlim(1)
  axes[0].set_ylabel('Accuracy Average', size=12, labelpad=10)
  axes[0].set_xlabel('Epoch', size=12, labelpad=10)
  axes[0].legend(loc='lower right', fontsize=10)
  axes[0].grid()

  axes[1].plot(train_loss, 'b', label='train errors')
  axes[1].plot(test_loss, 'r',label='test errors')
  axes[1].set_xlim(1)
#   axes[1].set_ylim(0, 1)
  axes[1].set_ylabel('Loss Average', size=12, labelpad=11)
  axes[1].set_xlabel('Epoch', size=12, labelpad=10)
  axes[1].legend(loc='best', fontsize=10)
  axes[1].grid()

  plt.show()

def plot_loss(train_loss, test_loss):
  _, axes = plt.subplots(1, 1, figsize=(5, 5))
  plt.suptitle("Loss plot for train and validation data", size=14)
  axes.plot(train_loss, 'b', label='train errors')
  axes.plot(test_loss, 'r',label='test errors')
  axes.set_xlim(1)
  # axes.set_ylim(0, 0.02)
  axes.set_ylabel('Loss Average', size=12, labelpad=11)
  axes.set_xlabel('Epoch', size=12, labelpad=10)
  axes.legend(loc='best', fontsize=10)
  axes.grid()
  plt.show()
