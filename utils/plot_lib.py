import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


def plt_statistic(history,ax,metric,title_below = False):
    ax.plot(history[metric])
    ax.plot(history['val_{}'.format(metric)])
    if title_below == True:
      ax.set_title('model {}'.format(metric),y = -0.01)
    else:
      ax.set_title('model {}'.format(metric))

    ax.set_ylabel(metric)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'], loc='upper right')


# helper function to plot confusion matrix 
def plot_confusion_matrix(ax ,y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                            ):
    """ (Copied from sklearn website)
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
            title = 'confusion matrix'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data


    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_ylim([-0.5, len(classes)-0.5])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return  cm