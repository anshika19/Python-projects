#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[3]:


#understanding the dataset
boston = load_boston()
print(boston.DESCR)


# In[4]:


#access data attributes
dataset = boston.data
for name, index in enumerate(boston.feature_names):
    print(index, name)


# In[5]:


#reshaping data
data = dataset[:,12].reshape(-1,1)


# In[6]:


#shape of data
np.shape(dataset)


# In[8]:


#target values
target = boston.target.reshape(-1,1)


# In[9]:


#shape of target
np.shape(target)


# In[31]:


#ensuring that matplotlib is working
get_ipython().system('matplotlib inline')
plt.scatter(data, target, color='purple')
plt.xlabel('Lower income population')
plt.ylabel('Cost of house')
plt.show()


# In[39]:


#regression
from sklearn.linear_model import LinearRegression

#creating regression model
reg =  LinearRegression()

#fitting the model
reg.fit(data, target)


# In[40]:


#prediction
pred= reg.predict(data)


# In[34]:


#ensuring that matplotlib is working
get_ipython().system('matplotlib incline')
plt.scatter(data, target, color= 'blue')
plt.plot(data, pred, color= 'purple')
plt.xlabel('Lower income population')
plt.ylabel('Cost of house')
plt.show()


# In[42]:


#circumventing curve issue using polynomial model
from sklearn.preprocessing import PolynomialFeatures

#to allow merging of models 
from sklearn.pipeline import make_pipeline


# In[44]:


model = make_pipeline(PolynomialFeatures(3), reg)


# In[45]:


model.fit(data, target)


# In[46]:


pred= model.predict(data)


# In[47]:


#ensuring that matplotlib is working
get_ipython().system('matplotlib incline')
plt.scatter(data, target, color= 'blue')
plt.plot(data, pred, color= 'purple')
plt.xlabel('Lower income population')
plt.ylabel('Cost of house')
plt.show()


# In[48]:


#r_2 metric
from sklearn.metrics import r2_score


# In[49]:


#predict
r2_score(pred, target)


# In[ ]:




