#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline


# In[2]:


seed=7
np.random.seed(seed)


# In[3]:


dataset = pd.read_csv('E:\My_Training\Bboks\iris.csv', header=None)
dataset=dataset.values

X= dataset[:, 0:4].astype(float)
y= dataset[:, 4]


# In[4]:


encoder=LabelEncoder()
encoder.fit(y)
coded_y= encoder.transform(y)
dummy_y= np_utils.to_categorical(coded_y)


# In[5]:


def baseline_model():
    model=Sequential()
    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(3, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[6]:


estimator=KerasClassifier(build_fn=baseline_model, epochs=150, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

results= cross_val_score(estimator, X, dummy_y, cv=kfold)


# In[7]:


print("%.2f%% (+- %.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:




