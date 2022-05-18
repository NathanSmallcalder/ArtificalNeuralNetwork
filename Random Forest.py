import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

data = pd.read_csv('AI.csv')

X = data.iloc[:,:-1].values
Y = data.iloc[:, -1].values
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
#Shuffles the data
np.random.shuffle(data.values)
#Initialize the training and test data
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
#Prints the shape of the data
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

forest = RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=1)
forest.fit(X_train,Y_train)

model = forest
print(model.score(X_train,Y_train))

cm = confusion_matrix(Y_test, model.predict(X_test))
print(cm)

TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

accuracy = (TP + TN) / (TP + TN + FN + FP)
print(accuracy)


