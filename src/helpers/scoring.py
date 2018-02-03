from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np

def accuracy(y_test, y_pred, classifier):
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    scores = (tn + tp) / np.sum((tn, fp, fn, tp))

    return cm, scores
