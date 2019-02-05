#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[2]:


seed = 7
np.random.seed(seed)


# In[3]:


dataset = np.loadtxt("E:\My_Training\Bboks\diabetes.csv", delimiter=",")
X = dataset[:, 0:8]
y = dataset[:, 8]


# In[4]:


model= Sequential()
model.add(Dense(12, input_dim = 8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=10)


# In[5]:


score=model.evaluate(X, y)


# In[6]:


print("%s: %.2f%%" %(model.metrics_names[1], score[1]*100))


# In[7]:


# Training using kfold


# In[8]:


kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, y):
        model= Sequential()
        model.add(Dense(12, input_dim = 8, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
        model.fit(X[train], y[train], epochs=150, batch_size=10, verbose=0)  
        scores=model.evaluate(X[test], y[test], verbose=0)  
        cvscores.append(scores[1]*100)


# In[9]:


print("%.2f%% (+- %.2f%%)" % (results.mean()*100, results.std()*100))


# In[11]:


# Training while wrapping with scikit learn


# In[12]:


def create_model():
    model= Sequential()
    model.add(Dense(12, input_dim = 8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
    return model


# In[13]:


model = KerasClassifier(build_fn=create_model, epochs = 150, batch_size=10,verbose=0)
kfold= StratifiedKFold(n_splits=10, shuffle=True,random_state=seed)

results=cross_val_score(model, X, y, cv=kfold)


# In[14]:


print("%.2f%% (+- %.2f%%)" % (results.mean()*100, results.std()*100))


# In[ ]:




