import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.datasets import mnist

(XTrain, yTrain), (XTest, yTest) = mnist.load_data()

fig = plt.figure()
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    ax.imshow(XTrain[i], cmap='gray', interpolation='none')
    ax.set_title(yTrain[i])
plt.tight_layout() 
#print(XTrain[0])
print(yTrain[0:9])

XTrain = XTrain.reshape(60000, 784)
XTest = XTest.reshape(10000, 784)
XTrain = XTrain/255
XTest  = XTest/255

YTrain = to_categorical(yTrain, 10)
YTest = to_categorical(yTest, 10)
print(YTrain[2])

myANN = Sequential()
myANN.add(Dense(80,input_dim=784,activation='relu'))
myANN.add(Dense(40,activation='relu'))
myANN.add(Dense(10,activation='sigmoid'))
myANN.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


myANN.fit(XTrain, YTrain, batch_size=24, epochs=20, verbose=True)

score = myANN.evaluate(XTest, YTest, verbose=False)
print('Test score:', score[0])
print('Test accuracy:', score[1])