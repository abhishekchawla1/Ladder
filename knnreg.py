#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'notebook')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\5. London Housing Data.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df['houses_sold'].values


# In[7]:


df.fillna(0,inplace=True)


# In[8]:


df


# In[9]:


df['date']=pd.to_datetime(df['date'])


# In[10]:


df


# In[11]:


df['year']=df['date'].dt.year


# In[12]:


df['month']=df['date'].dt.month


# In[14]:


len(df['area'].unique())


# In[15]:


len(df['code'].unique())


# In[16]:


df


# In[20]:


from sklearn.preprocessing import OrdinalEncoder


# In[21]:


o=OrdinalEncoder()


# In[22]:


e=o.fit_transform(df[['area']])


# In[23]:


p=pd.DataFrame(e,columns=['area'])


# In[24]:


p


# In[25]:


p['area'].unique()


# In[26]:


df['area']=p['area']


# In[27]:


df


# In[36]:


from sklearn.model_selection import train_test_split


# In[38]:


X=df[['area','houses_sold','no_of_crimes','year','month']]


# In[39]:


X


# In[40]:


y=df['average_price']


# In[41]:


from sklearn.neighbors import KNeighborsRegressor


# In[44]:


knn=KNeighborsRegressor(n_neighbors=5)


# In[45]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[46]:


X_train


# In[47]:


X_test


# In[49]:


knn.fit(X_train,y_train)


# In[50]:


knn.score(X_train,y_train)


# In[51]:


y_pred=knn.predict(X_test)


# In[53]:


from sklearn.metrics import r2_score


# In[54]:


r=r2_score(y_test,y_pred)


# In[55]:


r


# In[ ]:




