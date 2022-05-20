from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('AI.csv')

#Seperating the data
X = data.iloc[:,:-1].values
Y = data.iloc[:, -1].values

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

#Shuffles the data
np.random.shuffle(data.values)

#Initialize the tranning and test data
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

#Prints the shape of the data
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

#Initializes the neural network model
model = Sequential([
    Dense(32, activation='relu', input_shape=(9,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

#Compiles the model with sgd /////////
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
            batch_size=400,
            epochs=100,
            validation_data=(X_val,Y_val))

#Runs model
model.evaluate(X_test, Y_test)[0]

