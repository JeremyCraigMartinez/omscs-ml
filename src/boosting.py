# classifier did significantly worse when attempting to prune with
# max_depth and tweak parameters like leaf weight and random state

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

def boosting(X_train, X_test, y_train, y_test):
    classifier = AdaBoostClassifier()

    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    tmp_cm = confusion_matrix(y_test, y_pred)

    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
    accuracies.mean()
    accuracies.std()

    return tmp_cm
