#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


from sklearn.datasets import load_breast_cancer


# In[6]:


cancer=load_breast_cancer()
cancer


# In[10]:


df=pd.DataFrame(data=cancer.data,columns=cancer.feature_names)


# In[11]:


df


# In[12]:


df.shape


# In[13]:


cancer.keys()


# In[14]:


cancer.target


# In[15]:


cancer.DESCR


# In[16]:


len(cancer.feature_names)


# In[17]:


df2=pd.DataFrame(cancer.target,columns=['Target'])


# In[18]:


df2


# In[24]:


df.shape


# In[25]:


df2.shape


# In[33]:


d=pd.concat([df,df2],axis=1)


# In[34]:


d.shape


# In[35]:


d['Target'].unique()


# In[36]:


d.isnull().sum()


# In[37]:


d


# In[40]:


d['Target'].value_counts()


# In[41]:


target=pd.Series([357,212],index=['malignant','benign'])


# In[42]:


target


# In[52]:


X=df
y=df2


# In[53]:


X.shape


# In[54]:


y.shape


# In[55]:


from sklearn.model_selection import train_test_split


# In[58]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[59]:


X_train.shape


# In[60]:


X_test.shape


# In[61]:


y_train.shape


# In[62]:


y_test.shape


# In[64]:


from sklearn.neighbors import KNeighborsClassifier


# In[67]:


knn=KNeighborsClassifier(n_neighbors=1)


# In[69]:


model=knn.fit(X_train,y_train)


# In[71]:


len(model.predict(X_test))


# In[72]:


y_test.shape


# In[81]:


x=d.mean()[:-1].values.reshape(1,-1)


# In[82]:


y_pred=model.predict(X_test)


# In[83]:


y_pred.shape


# In[84]:


model.predict(x)


# In[85]:


y_pred


# In[86]:


from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix


# In[87]:


a=accuracy_score(y_test,y_pred)


# In[88]:


p=precision_score(y_test,y_pred)
r=recall_score(y_test,y_pred)
f=f1_score(y_test,y_pred)
c=confusion_matrix(y_test,y_pred)


# In[89]:


a


# In[90]:


p


# In[91]:


r


# In[92]:


f


# In[93]:


c


# In[94]:


knn.score(X_test,y_test)


# In[95]:


model.score(X_test,y_test)

