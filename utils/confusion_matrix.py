import matplotlib.pyplot as plt
import pandas as pd
import operator
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report
import numpy as np


def save_cr_and_cm(index_to_entity, list_of_y_real, list_of_pred_tags, cr_save_path="classification_report.csv", cm_save_path="confusion_matrix.png"):
    """ print classification report and confusion matrix """

    # target_names = val_dl.dataset.ner_to_index.keys()
    #sorted_ner_to_index = sorted(val_dl.dataset.ner_to_index.items(), # # ey=operator.itemgetter(1))
    sorted_ner_to_index=sorted(index_to_entity.items())
    print('sorted_ner_to_index')
    print(sorted_ner_to_index)
    target_names = []
    for index, tag in sorted_ner_to_index:
        if tag in ['[CLS]', '[SEP]', '[PAD]', '[MASK]', 'O']:
            continue
        else:
            target_names.append(tag)

    label_index_to_print = list(range(5, 25))  # ner label indice except '[CLS]', '[SEP]', '[PAD]', '[MASK]' and 'O' tag
    print(classification_report(y_true=list_of_y_real, 
                                y_pred=list_of_pred_tags, 
                                target_names=target_names, 
                                labels=label_index_to_print,
                                digits=4))
    
    cr_dict = classification_report(y_true=list_of_y_real, 
                                    y_pred=list_of_pred_tags, 
                                    target_names=target_names, 
                                    labels=label_index_to_print, 
                                    digits=4, output_dict=True)
    
    df = pd.DataFrame(cr_dict).transpose()
    df.to_csv(cr_save_path)
    # np.set_printoptions(precision=2)
    
    plot_confusion_matrix(y_true=list_of_y_real, 
                          y_pred=list_of_pred_tags, 
                          classes=target_names, 
                          labels=label_index_to_print, 
                          normalize=False, 
                          title='Confusion matrix, without normalization')
    
    plt.savefig(cm_save_path)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, labels,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # --- plot 크기 조절 --- #
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = [20, 20]  # plot 크기
    plt.rcParams.update({'font.size': 10})
    # --- plot 크기 조절 --- #

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # --- bar 크기 조절 --- #
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # --- bar 크기 조절 --- #
    # ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax