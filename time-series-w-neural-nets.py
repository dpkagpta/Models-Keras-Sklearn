#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


seed = 7
np.random.seed(seed)


# In[3]:


df = pd.read_csv(r'E:\My_Training\Bboks\data\international-airline-passengers.csv', skipfooter =3, engine='python',            usecols = [1])
dataset = df.values.astype(float)
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0: train_size, :], dataset[0: test_size, :]


# In[4]:


plt.plot(df)
plt.show()


# In[5]:


# By using multilayer perceptrons regression


# In[6]:


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i: (i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)
        
look_back = 1
trainX, trainY = create_dataset(dataset, look_back=1)
testX, testY = create_dataset(dataset, look_back=1)

model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs = 200, batch_size = 2, verbose=2)


# In[7]:


train_score = model.evaluate(trainX, trainY, verbose= 0)
print('Training Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
test_score = model.evaluate(testX, testY, verbose= 0)
print('Testing Score: %.2f MSE (%.2f RMSE)' % (test_score, math.sqrt(test_score)))

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[look_back:len(testPredict)+look_back, :] = testPredict

plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[8]:


# using window approch  


# In[9]:


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i: (i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return np.array(dataX), np.array(dataY)
        
look_back = 3
trainX, trainY = create_dataset(dataset, look_back=3)
testX, testY = create_dataset(dataset, look_back=3)

model = Sequential()
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs = 200, batch_size = 2, verbose=2)


# In[10]:


train_score = model.evaluate(trainX, trainY, verbose= 0)
print('Training Score: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
test_score = model.evaluate(testX, testY, verbose= 0)
print('Testing Score: %.2f MSE (%.2f RMSE)' % (test_score, math.sqrt(test_score)))

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[look_back:len(testPredict)+look_back, :] = testPredict

plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[ ]:




