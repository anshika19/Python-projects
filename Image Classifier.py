#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().system('matplotlib inline')


# In[2]:


data = pd.read_csv('mnist_test.csv')


# In[3]:


# column heads
data.head()


# In[17]:


#extracting data from set 
a = data.iloc[0,1:].values


# In[18]:


#reshaping the extracted data
a = a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[6]:


#preparing the data
#seperating labels and data values
df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]


# In[7]:


#creating test and train sizes
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.2, random_state=4)


# In[8]:


#check data
y_train.head()


# In[20]:


#call rf classifier
rf = RandomForestClassifier(n_estimators= 100)


# In[21]:


#fit the model
rf.fit(x_train, y_train)


# In[22]:


#prediction on test data
pred = rf.predict(x_test)
pred


# In[26]:


#check prediction accuracy
s = y_test.values

#calculate no of correctly predicted values
count = 0
for i in range (len(pred)):
    if pred[i] == s[i]:
        count = count+1


# In[27]:


count


# In[28]:


#total values that the prediction code was run on
len(pred)


# In[29]:


#accuracy value
1885/2000


# In[ ]:




