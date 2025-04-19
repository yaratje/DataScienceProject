from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def accuracy_metric(actual, predicted):
    
    # Ex.2.4 your code here

    #(Num Correct Predictions) / ( Num Total Predictions)
    accuracy_value = accuracy_score(actual, predicted)
    #TP / (TP + FN)
    recall = recall_score(actual,predicted,average='macro')
    #TP / (TP + FP)
    precision = precision_score(actual,predicted,average='macro')
    #more usefull for imbalanced data which with the cifar is not the case but meh do it anyway.
    fscore = f1_score(actual,predicted,average='macro')
    
    return accuracy_value, recall, precision, fscore

def kfold(model, features, labels):
    k_folds = [2, 5, 10]

    for k in k_folds:
        scores = cross_val_score(model, features, labels, cv=k, scoring='accuracy')
    
        print(f"{k}-Fold Cross-Validation: Mean Accuracy={scores.mean():.4f}, Std Dev={scores.std():.4f}")


def confusion(actual, predicted,fold=None, cmap="Blues"):
    class_names = np.unique(actual)
    matrix = confusion_matrix(predicted, actual)
    fig, ax = plt.subplots(figsize=(len(class_names)*0.6, len(class_names)*0.6))
    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=f'Confusion Matrix{" (fold "+str(fold+1)+")" if fold is not None else ""}'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig("confusion_matrix")