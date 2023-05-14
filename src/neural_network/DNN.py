from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.models import Sequential


def DNN(X_train_scaled, y_train, X_test_scaled, y_test, nb_hidden_layer=5, nb_neurals=500, dropout=0.2, patience_stopping=25, epochs=100):
    """ Deep Neural Network model """

    # Initialize the RNN
    regressor = Sequential()

    # First layer
    regressor.add(Dense(nb_neurals, input_shape=(X_train_scaled.shape[1],), activation="relu"))

    # Hidden layers
    for i in range(1, nb_hidden_layer):
        regressor.add(Dense(int(nb_neurals), activation="relu"))
        regressor.add(Dropout(dropout))

    # Output layer : autant qu'il y a de nombres de sorties : une seule dans notre cas
    try :
        regressor.add(Dense(y_train.shape[1], activation="linear"))
    
    except IndexError:
        regressor.add(Dense(1, activation="linear"))

    # compile
    regressor.compile(optimizer="adam", loss="mean_squared_error")
        
    # Evite l'overfitting
    early_stop = EarlyStopping(patience=patience_stopping, verbose=1) # L'algo s'arrête si l'erreur augmente sur 10 epochs

    # training
    regressor.fit(X_train_scaled, y_train, epochs=epochs, validation_data=(X_test_scaled, y_test), callbacks=[early_stop])

    return regressor
