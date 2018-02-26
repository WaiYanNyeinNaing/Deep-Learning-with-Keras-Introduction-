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



#one_hot_encode function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels),labels]=1
    return one_hot_encode
    

#Read the dataset
def read_dataset():
    data = pd.read_csv("iris.csv")
    #print(len(data.columns)) #number of features
    x = data[data.columns[0:4]].values
    y = data[data.columns[[4]]]
    #Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    y = one_hot_encode(y)
    return (x,y)


#Read Input/Target from dataset
X,Y = read_dataset()
X,Y = shuffle(X,Y,random_state=1)       
#Convert dataset into train and test part (10% data as test set)
train_x , test_x , train_y , test_y = train_test_split(X,Y,test_size=0.1,random_state=415)
#Check the shape of input and target data
print(train_x.shape)
print(train_y.shape)


# 1. define the network
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))   #Input features = 4 , hidden1 = 12
model.add(Dense(3, activation='sigmoid'))              #output class = 3  

# 2. compile the network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3. fit the network
history = model.fit(train_x, train_y, epochs=1000, batch_size=10)

# 4. evaluate the network for Training
loss_1, accuracy_1 = model.evaluate(X, Y)
print("\nLoss: %.2f, Train Accuracy: %.2f%%" % (loss_1, accuracy_1*100))

# 5. make predictions for Testing
predictions = model.predict(test_x)
predictions = numpy.round(predictions)
print("Prediction output is  %s " % (predictions))
print("Actual output is  %s " %  (test_y))
loss, accuracy = model.evaluate(test_x, test_y)
print("\nLoss: %.2f, Train Accuracy: %.2f%%" % (loss_1, accuracy_1*100))
print("\nLoss: %.2f, Test Accuracy: %.2f%%" % (loss, accuracy*100))



