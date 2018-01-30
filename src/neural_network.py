# Import processed data
from nl_preprocessing import get_train_test_set

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

def ann(X_train, X_test, y_train, y_test):
    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 1500))

    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    # Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

    # Part 3 - Making the predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    return cm

if __name__ == '__main__':
    split_data_set = get_train_test_set()
    cm = ann(*split_data_set)
    print(cm)
