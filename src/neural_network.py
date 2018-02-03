# Input dimension needs to match data, necessary to pass as an argument per basis
# a unique dimensionality of output space seems to make a significant impact for
#   input later, however, for hidden layer, it is a little more touchy and seems
#   to negatively impact each dataset

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from helpers.scoring import metrics

def ann(X_train,
        X_test,
        y_train,
        y_test,
        input_dim=None,
        units=500):
    classifier = Sequential()

    #Input layer
    classifier.add(Dense(units=units,
                         input_dim=input_dim,
                         kernel_initializer='uniform',
                         activation='relu'))
    classifier.add(Dropout(0.5))

    #Hidden layer 1
    classifier.add(Dense(units=200,
                         kernel_initializer='uniform',
                         activation='relu'))
    classifier.add(Dropout(0.5))

    #Output layer
    classifier.add(Dense(units=1,
                         kernel_initializer='uniform',
                         activation='sigmoid'))

    # Defining how to measure performance
    classifier.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

    classifier.fit(x=X_train,
                   y=y_train,
                   validation_split=0.2,
                   epochs=20,
                   batch_size=200,
                   verbose=0)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    return metrics(y_test, y_pred)
