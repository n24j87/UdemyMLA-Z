#Data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encode_X_1 = LabelEncoder()
X[:, 1] = label_encode_X_1.fit_transform(X[:, 1])
label_encode_X_2 = LabelEncoder()
X[:, 2] = label_encode_X_2.fit_transform(X[:, 2])
onehotendocder = OneHotEncoder(categorical_features = [1])
X = onehotendocder.fit_transform(X).toarray()
#Remove dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN
# Importing keras library and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN model
classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(6, activation='relu', input_shape=(11,)))

# Add second hidden layer
classifier.add(Dense(6, activation='relu'))

# Adding the output layer
classifier.add(Dense(1, activation='sigmoid'))

# Compiling the model
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting ANN to training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - making the predictions and evaluating the models

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)