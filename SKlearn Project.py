#!/usr/bin/env python
# coding: utf-8

#   # How SKlearn Works Generally
#   #### Data -> Model -> Predictions
#   #### More specifically we usually split the data up into two parts X and Y. X represents everything being used to make the perdiction and Y represents the prediction we are interested in making.

# In[2]:


## Loading Boston Housing Dataset
from sklearn.datasets import fetch_california_housing


# In[4]:


# Variables that we can use in model
X,y = fetch_california_housing(return_X_y=True)


# In[11]:


fetch_california_housing()


# # What does a model do?
# #### The model is created as python object and then the model begins to learn at the .fit step

# In[14]:


# Loading up k nearest neighbor model

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression


# In[8]:


mod = KNeighborsRegressor()


# In[12]:


mod.fit(X,y)


# In[13]:


mod.predict(X)


# In[15]:


# Loading up Linear model
from sklearn.linear_model import LinearRegression
moda = LinearRegression()


# In[16]:


moda.fit(X,y)


# In[18]:


moda.predict(X)


# In[ ]:




