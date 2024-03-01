from cleanData import preprocess_data
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from keras import utils, backend, Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shap

file = "/Users/danteamicarella/Downloads/student-mat_modified (1)-1.csv"
df = preprocess_data(file)

columns_drop = ["Performance", "sex", "nursery"]

X = df.drop(columns_drop, axis = 1)
Y = df["Performance"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

# One layer 
# model = models.Sequential(name="Perceptron", layers=[
#     layers.Dense(             #a fully connected layer
#           name="dense",
#           input_dim=3,        #with 3 features as the input
#           units=1,            #and 1 node because we want 1 output
#           activation='linear' #f(x)=x
#     )
# ])
# model.summary()


