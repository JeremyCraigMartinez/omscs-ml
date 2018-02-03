from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from sklearn.metrics import confusion_matrix

def ann(X_train, X_test, y_train, y_test, input_dim=1500):
    model = Sequential()

    #Input layer
    model.add(Dense(units=500,
                    input_dim=input_dim,
                    kernel_initializer='uniform',
                    activation='relu'))
    model.add(Dropout(0.5))

    #Hidden layer 1
    model.add(Dense(units=200,
                    kernel_initializer='uniform',
                    activation='relu'))
    model.add(Dropout(0.5))

    #Output layer
    model.add(Dense(units=1,
                    kernel_initializer='uniform',
                    activation='sigmoid'))

    print(model.summary()) #for showing the structure and parameters

    # Defining how to measure performance
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    model.fit(x=X_train, y=y_train,
              validation_split=0.2, epochs=20,
              batch_size=200, verbose=2)

    # Predicting the Test set results
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    # Making the Confusion Matrix
    return confusion_matrix(y_test, y_pred)
