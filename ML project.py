#!/usr/bin/env python
# coding: utf-8

# In[4]:


import sys
print('Python: {}'.format(sys.version))
import scipy
print('Scipy: {}'.format(scipy.__version__))
import numpy
print('Numpy: {}'.format(numpy.__version__))
import matplotlib
print('Matplotlib: {}'.format(matplotlib.__version__))
import pandas
print('Pandas: {}'.format(pandas.__version__))
import sklearn
print('Sklearn: {}'.format(sklearn.__version__))


# In[5]:


#iris flower dataset
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier


# In[6]:


#loading the data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names= ["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset= read_csv(url, names=names)


# In[7]:


#dimensions of dataset
print(dataset.shape)


# In[8]:


#take a peak at data
print(dataset.head(20))


# In[9]:


#statistical summary
print((dataset.describe()))


# In[10]:


#class distribution
print(dataset.groupby("class").size())


# In[11]:


#univariate plots- box and whisker plots
dataset.plot(kind= 'box', subplots= True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# In[12]:


#histogram
dataset.hist()
pyplot.show()


# In[13]:


#multivariate plots
scatter_matrix(dataset)
pyplot.show()


# In[14]:


#creating validation dataset
#splitting dataset (train is always more than test)
array = dataset.values
x= array[:, 0:4]
y= array[:,4]
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= 0.2, random_state=1)


# In[15]:


#logistic regression
#linear discriminant analysis
#k nearest neighbors
#clssification and regression trees
#gaussian naive bayes
#support vectors machine

#building models
models= []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma= 'auto')))


# In[16]:


#evaluating models
results= []
names= []
for name, model in models:
    kfold= StratifiedKFold(n_splits= 10, random_state=1, shuffle= True)
    cv_results= cross_val_score(model, x_train, y_train, cv= kfold , scoring= 'accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' %(name, cv_results.mean(), cv_results.std()))


# In[17]:


#compare our models
pyplot.boxplot(results, labels= names)
pyplot.title('Algorithm Comparison')
pyplot.show()


# In[21]:


#make prediction on svm
model = SVC(gamma = 'auto')
model.fit(x_train, y_train)
predictions= model.predict(x_test)


# In[22]:


#evaluate predictions
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:




