from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from helpers.scoring import metrics

def knn(X_train, X_test, y_train, y_test, n_neighbors=3):
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting K-NN to the Training set
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean', weights='distance')
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    return metrics(y_test, y_pred)
