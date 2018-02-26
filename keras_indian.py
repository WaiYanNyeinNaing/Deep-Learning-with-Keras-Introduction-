# Sample Multilayer Perceptron Neural Network in Keras
#Import Library
from keras.models import Sequential
from keras.layers import Dense
import numpy
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# load and prepare the dataset
dataset = numpy.loadtxt("indian.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]    
X,Y = shuffle(X,Y,random_state=1) 

#Convert dataset into train and test part (20% data as test set)
train_x , test_x , train_y , test_y = train_test_split(X,Y,test_size=0.2,random_state=415)
print(train_x.shape)
print(train_y.shape)

# 1. define the network
model = Sequential()
model.add(Dense(30, input_dim=8, activation='relu'))
model.add(Dense(12, input_dim=30, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 2. compile the network
model.compile(loss='binary_crossentropy', optimizer= 'adam' , metrics=['accuracy'])

# 3. fit the network
history = model.fit(train_x, train_y, epochs=1000, batch_size=32)

# 4. evaluate the network for Training 
loss_1, accuracy_1 = model.evaluate(train_x, train_y)
print("\nLoss: %.2f, Train Accuracy: %.2f%%" % (loss_1, accuracy_1*100))

# 5. make predictions for Testing
predictions = model.predict(test_x)
predictions = numpy.round(predictions)
loss, accuracy = model.evaluate(test_x, test_y)

#Print Accuracy 
print("\nLoss: %.2f, Train Accuracy: %.2f%%" % (loss_1, accuracy_1*100))
print("\nLoss: %.2f, Test Accuracy: %.2f%%" % (loss, accuracy*100))


