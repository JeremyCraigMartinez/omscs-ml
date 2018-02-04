from sklearn.metrics import confusion_matrix
import numpy as np

def metrics(y_test, y_pred):
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tn + tp) / np.sum((tn, fp, fn, tp))
    precision = (tp) / np.sum((tp, fp))
    specifity = (tn) / np.sum((tn, fp))
    tpr = (tp) / np.sum((tp, fn)) # recall
    fpr = (fp) / np.sum((tn, fp))

    return cm, accuracy, precision, specifity, tpr, fpr
