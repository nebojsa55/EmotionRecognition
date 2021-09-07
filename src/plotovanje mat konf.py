# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 07:39:05 2021

@author: HP
"""

import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm)/np.sum(cm).astype('float')
    prec = cm[1,1]/(cm[1,1] + cm[0,1])
    recall = cm[1,1]/(cm[1,1] + cm[1,0])
#    prec = cm[0,0]/(cm[0,0] + cm[1,0])
#    recall = cm[0,0]/(cm[0,0] + cm[0,1])
    f1_score = 2*(prec*recall/(prec + recall))

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; F1 score={:0.4f}'.format(accuracy, f1_score))
    plt.show()

# Dominance - SVM(C = 1, 'linear'), acc = 71% +- 3%
cm = np.array([[201, 223],[146, 710]]) # acc0 = 47.4%   acc1 = 82.9%
plot_confusion_matrix(cm, [0, 1], 'Dominance')

# Liking - najbolji je linearan sa C = 0.01 ALI onda se overfituje i sve klasifikuje u klasu 1
# zato sam koristila C = 1, acc = 71% +- 2%
cm = np.array([[155, 235],[138, 752]]) # acc0 = 39.7%   acc1 = 84.5% 
plot_confusion_matrix(cm, [0, 1], 'Liking')

# Arousal C = 1 'linear'  acc = 67% +- 5%
cm = np.array([[215, 247],[174, 644]]) # acc0 = 46.5%   acc1 = 78.7%
plot_confusion_matrix(cm, [0, 1], 'Arousal')

# Valence  C = 1  'linear' acc = 66% +- 3%
cm = np.array([[220, 252],[181, 627]]) # acc0 = 46.6%   acc1 = 77.6%
plot_confusion_matrix(cm, [0, 1], 'Valence')