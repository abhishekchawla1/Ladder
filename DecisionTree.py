#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris


# In[3]:


iris=load_iris()


# In[4]:


X=iris.data


# In[5]:


y=iris.target


# In[6]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[9]:


X_train


# In[10]:


X_test


# In[11]:


y_train


# In[12]:


y_test


# In[13]:


from sklearn.tree import DecisionTreeClassifier


# In[14]:


dt=DecisionTreeClassifier()


# In[15]:


dt.fit(X_train,y_train)


# In[16]:


y_pred=dt.predict(X_test)


# In[17]:


from sklearn.metrics import accuracy_score


# In[18]:


a=accuracy_score(y_test,y_pred)


# In[19]:


a


# In[20]:


from sklearn.tree import plot_tree


# In[22]:


from matplotlib.pylab import rcParams


# In[24]:


rcParams['figure.figsize']=80,50


# In[27]:


plot_tree(dt)

