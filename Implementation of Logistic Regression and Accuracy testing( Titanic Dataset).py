#!/usr/bin/env python
# coding: utf-8

# In[48]:


## Collecting the Data : Step 1
import pandas as pd 
# pandas library is used for data analysis
import numpy as np
# numpy is a library in python which stands for numerical python and is widely used for any scientific computation
import seaborn as sns
# seaborn is a library for statistical plotting
import matplotlib.pyplot as plt
# matplotlib is used for plotting


import math
# to calculate basic mathematics functions

titanic_data = pd.read_csv("titanic.csv")
titanic_data.head(10)


# In[49]:


print("# of passengers in data "+ str(len(titanic_data.index)))


# ## Analyzing the Data : Step 2

# In[50]:


sns.countplot(x = "survived", data = titanic_data)


# In[51]:


sns.countplot(x = "survived", hue = "sex" ,data = titanic_data)


# In[14]:


sns.countplot(x = "survived",hue = "pclass" ,data = titanic_data)


# In[52]:


titanic_data["age"].plot.hist()


# In[53]:


titanic_data["fare"].plot.hist()


# In[21]:


titanic_data["age"].plot.hist(bins=20, figsize = (10,5)) 
# to make it more clear we add bin and figsize arguments


# In[25]:


titanic_data.info()


# ## Data Wrangling : Step 3
# 

# In[54]:


titanic_data.isnull()
# False means value is not null and True means value is null. But we need to count how many null values in each column for better understanding.


# In[55]:


titanic_data.isnull().sum()


# In[56]:


sns.heatmap(titanic_data.isnull(),yticklabels = False, cmap = "Reds" )


# In[57]:


## To remove column cabin from data as there are so many null values in that column
titanic_data.head(5)


# In[58]:


titanic_data.drop("cabin", axis =1, inplace = True)


# In[59]:


## To check whether the column cabin has been dropped from dataset or not
titanic_data.head(5)


# In[61]:


titanic_data.drop("body", axis =1, inplace = True)


# In[67]:


titanic_data.drop("boat", axis =1, inplace = True)
titanic_data.head(5)


# In[69]:


titanic_data.drop("home.dest", axis =1, inplace = True)
# To confirm na values has been dropped from dataset


# In[70]:


titanic_data.head(5)


# In[71]:


sns.heatmap(titanic_data.isnull(),yticklabels = False, cmap = "Reds" )


# In[72]:


# to drop all null values from dataset
titanic_data.dropna(inplace=True)


# In[73]:


sns.heatmap(titanic_data.isnull(),yticklabels = False, cmap = "Reds" )


# In[74]:


titanic_data.isnull().sum()


# In[75]:


# for logistic regression we need values not to be in string format so we have to create dummi variables for those columns
sex = pd.get_dummies(titanic_data["sex"],drop_first = True)


# In[76]:


sex.head(5)


# In[77]:


embark = pd.get_dummies(titanic_data["embarked"],drop_first = True)
embark.head(5)


# In[81]:


pcl = pd.get_dummies(titanic_data["pclass"],drop_first = True)
pcl.head(5)


# In[82]:


titanic_data=pd.concat([titanic_data,sex,embark,pcl], axis=1)
titanic_data.head(5)


# In[83]:


titanic_data.drop(["name","sex","embarked","ticket"], axis = 1, inplace = True )


# In[84]:


titanic_data.head(5)


# In[85]:


titanic_data.drop("pclass", axis = 1, inplace = True )


# In[86]:


titanic_data.head(5)


# ## Training data : Step 4

# In[92]:


# independent variable is X which has all features except survived
X = titanic_data.drop("survived",axis = 1) 
# dependent variable is Y which we will be predicting depending on X
y = titanic_data["survived"]


# In[93]:


from sklearn.model_selection import train_test_split


# In[94]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)


# In[95]:


from sklearn.linear_model import LogisticRegression


# In[96]:


logmodel= LogisticRegression()


# In[97]:


logmodel.fit(X_train,y_train)


# In[98]:


predictions = logmodel.predict(X_test)


# ## Accuracy Test : Step 5
# 
# 

# In[105]:


# Finding accuracy using classification_report
from sklearn.metrics import classification_report
classification_report(y_test,predictions)


# In[106]:


# Finding accuracy using confusion matrix
from sklearn.metrics import confusion_matrix


# In[102]:


confusion_matrix(y_test,predictions)


# In[103]:


# Finding accuracyusing accuracy score
from sklearn.metrics import accuracy_score


# In[104]:


accuracy_score(y_test,predictions)


# In[ ]:




