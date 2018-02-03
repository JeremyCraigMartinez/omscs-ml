from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

def accuracy(X_train, y_train, y_test, y_pred, classifier):
    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    mean = accuracies.mean()

    return cm, mean
