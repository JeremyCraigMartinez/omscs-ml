from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from nl_preprocessing import get_train_test_set

def knn(X_train, X_test, y_train, y_test):
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting K-NN to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    return confusion_matrix(y_test, y_pred)

if __name__ == '__main__':
    split_data_set = get_train_test_set()
    cm = knn(*split_data_set)
    print(cm)
