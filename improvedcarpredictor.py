#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df=pd.read_csv(r'C:\Users\ASUS\Downloads\quikr_car.csv')


# In[3]:


df.head(2)


# In[4]:


df.info()


# In[5]:


df['year'].unique()


# In[6]:


df=df[df['year'].str.isnumeric()]


# In[7]:


df.shape


# In[8]:


df.year.unique()


# In[9]:


df['year']=df['year'].astype(int)


# In[10]:


df['Price'].unique()


# In[11]:


df=df[df['Price']!='Ask For Price']


# In[12]:


df['Price']=df['Price'].apply(lambda x: int(x.replace(',','')))


# In[13]:


df.Price.dtypes


# In[14]:


df.head(2)


# In[15]:


df['kms_driven'].unique()


# In[16]:


df=df[df['kms_driven']!='Petrol']


# In[17]:


df['kms_driven']=df['kms_driven'].apply(lambda x: int(x.split()[0].replace(',','')))


# In[18]:


df['kms_driven'].info()


# In[19]:


df


# In[20]:


df.isnull().sum()


# In[21]:


df.dropna(inplace=True)


# In[22]:


df.shape


# In[23]:


df=pd.get_dummies(df,columns=['fuel_type'])


# In[24]:


df


# In[25]:


from sklearn.preprocessing import OrdinalEncoder


# In[26]:


o=OrdinalEncoder()


# In[27]:


df['company']=o.fit_transform(df['company'].values.reshape(-1,1))


# In[28]:


df


# In[29]:


df['name']=df['name'].str.split().str.slice(start=0,stop=3).str.join(' ')


# In[30]:


df


# In[31]:


df.describe()


# In[32]:


df=df[df['Price']<6000000]


# In[33]:


df


# In[34]:


o.categories_


# In[35]:


sns.boxplot(x='company',y='Price',data=df)
plt.xticks(rotation=60)
plt.show


# In[36]:


sns.swarmplot(x='year',y='Price',data=df,hue='company')
plt.xticks(rotation=90)
plt.show()


# In[37]:


sns.distplot(df['kms_driven'])


# In[38]:


plt.figure(figsize=(20,5))
sns.boxplot(x='fuel_type_Petrol',y='Price',data=df)


# In[39]:


X=df[['name','company','year','kms_driven','fuel_type_Petrol','fuel_type_LPG','fuel_type_Diesel']]


# In[40]:


y=df['Price']


# In[41]:


X


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)


# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


from sklearn.metrics import r2_score


# In[46]:


from sklearn.preprocessing import OneHotEncoder


# In[47]:


o=OneHotEncoder(sparse=False)


# In[48]:


o.fit(X[['name']])


# In[49]:


l=LinearRegression()


# In[50]:


l


# In[51]:


X_train['name']=o.transform(X_train['name'].values.reshape(-1,1))


# In[52]:


X_test['name']=o.transform(X_test['name'].values.reshape(-1,1))


# In[55]:


l.fit(X_train,y_train)


# In[56]:


p=l.predict(X_test)


# In[58]:


r=r2_score(y_test,p)


# In[ ]:




