#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


# In[2]:


# Loading the dataset


# In[3]:


dataset=pd.read_csv(r'E:\My_Training\Bboks\sonar.all-data.csv', header=None)


# In[4]:


dataset=dataset.values
X=dataset[:, 0:60].astype(float)
y=dataset[:,60]


# In[5]:


seed=7
np.random.seed(seed)


# In[6]:


encoder=LabelEncoder()
encoder.fit(y)
coded_y=encoder.transform(y)


# In[7]:


## A general neural net


# In[8]:


def create_baseline():
    model=Sequential()
    model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))    
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    sgd=SGD(lr=0.01, momentum=0.8, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

estimators=[]
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=300, batch_size=16, verbose=0)))
pipeline=Pipeline(estimators)
kfold=KFold(n_splits=10, shuffle=True, random_state=seed)

results=cross_val_score(pipeline, X, coded_y, cv=kfold)

print('Baseline: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))


# In[9]:


# using Dropout with visible layer


# In[13]:


def create_baseline():
    model=Sequential()
    model.add(Dropout(0.2, input_shape=(60,)))
    model.add(Dense(60, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))    
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    sgd=SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

estimators=[]
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=500, batch_size=16, verbose=0)))
pipeline=Pipeline(estimators)
kfold=KFold(n_splits=10, shuffle=True, random_state=seed)

results=cross_val_score(pipeline, X, coded_y, cv=kfold)

print('Baseline: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))


# In[14]:


# using Dropout with hidden layers


# In[15]:


def create_baseline():
    model=Sequential()
    model.add(Dense(60, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))  
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    sgd=SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

estimators=[]
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=500, batch_size=16, verbose=0)))
pipeline=Pipeline(estimators)
kfold=KFold(n_splits=10, shuffle=True, random_state=seed)

results=cross_val_score(pipeline, X, coded_y, cv=kfold)

print('Baseline: %.2f%% (%.2f%%)' % (results.mean()*100, results.std()*100))


# In[ ]:




