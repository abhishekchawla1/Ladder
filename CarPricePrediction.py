#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\quikr_car.csv")


# In[4]:


df


# In[5]:


df.columns


# In[6]:


df.fuel_type.value_counts()


# In[7]:


df.year.unique()


# In[8]:


df['company'].nunique()


# In[9]:


df.duplicated().any()


# In[10]:


df.drop_duplicates(inplace=True)


# In[11]:


df.info()


# In[12]:


df.isnull().sum()


# In[13]:


df=pd.get_dummies(df,columns=['fuel_type'])


# In[14]:


df.kms_driven.unique()


# In[15]:


df=df[df['kms_driven']!='Petrol']


# In[16]:


df.kms_driven.unique()


# In[17]:


df.isnull().sum()/len(df)


# In[18]:


df.describe()


# In[19]:


df.dtypes


# In[20]:


df.sample(5)


# In[21]:


df.dtypes


# In[22]:


df.isnull().sum()


# In[23]:


df.dropna(inplace=True)


# In[24]:


df


# In[25]:


df['kms_driven']=df['kms_driven'].apply(lambda x : x.split()[0])


# In[26]:


df


# In[27]:


df['kms_driven']=df['kms_driven'].apply(lambda x: x.replace(',',''))


# In[28]:


df


# In[29]:


df=df[df['year'].str.isnumeric()]


# In[30]:


df['kms_driven']=df['kms_driven'].apply(lambda x: int(x))


# In[31]:


df.dtypes


# In[32]:


df['Price'].unique()


# In[33]:


df[df['Price']=='Ask For Price']


# In[34]:


sns.boxplot(df['kms_driven'])


# In[35]:


df[df['kms_driven']==df['kms_driven'].max()]


# In[36]:


import scipy.stats as stats


# In[37]:


stats.probplot(df['kms_driven'],dist='norm',plot=plt)


# In[38]:


sns.distplot(df['kms_driven'])


# In[39]:


df['kms_driven'].skew()


# In[40]:


from sklearn.preprocessing import FunctionTransformer


# In[41]:


f=FunctionTransformer(func=np.log1p)


# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


df['Price']


# In[44]:


df['Price']=df['Price'].replace('Ask For Price',np.NaN)


# In[45]:


df


# In[46]:


df['Price']=df['Price'].replace(',','')


# In[47]:


df


# In[48]:


df['Price'].dtype


# In[49]:


df['Price'] = df['Price'].apply(lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)


# In[50]:


df


# In[51]:


df['Price'].isnull().sum()/len(df)


# In[52]:


df.dropna(inplace=True)


# In[53]:


df.shape


# In[54]:


df.dtypes


# In[55]:


df.describe()


# In[56]:


sns.boxplot(df['Price'])


# In[57]:


df[df['Price']>6000000]


# In[58]:


df=df[df['Price']<6000000]


# In[59]:


df.shape


# In[ ]:





# In[ ]:





# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


X=df[['name','company','year','kms_driven','fuel_type_Diesel','fuel_type_Petrol','fuel_type_LPG']]
y=df['Price']
X_train,X_test,y_train,y_test=train_test_split(X,y)


# In[62]:


f.fit(X_train)


# In[63]:


X_norm=f.transform(X_train['kms_driven'])


# In[64]:


X_norm


# In[65]:


sns.distplot(X_norm)


# In[66]:


df['name'].nunique()


# In[67]:


df['company'].nunique()


# In[68]:


from sklearn.preprocessing import OrdinalEncoder


# In[69]:


o=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)


# In[70]:


X_train['company']=o.fit_transform(X_train['company'].values.reshape(-1,1))


# In[71]:


X_test['company']=o.transform(X_test['company'].values.reshape(-1,1))


# In[72]:


X_test


# In[73]:


X_train


# In[74]:


X_test


# In[75]:


df['year'].unique()


# In[76]:


X_train['year']=X_train['year'].apply(lambda x : int(x))


# In[77]:


X_train.dtypes


# In[78]:


X_test['year']=X_test['year'].apply(lambda x: int(x))


# In[79]:


X_test


# In[80]:


X_train.drop(columns=['name'],inplace=True)


# In[81]:


X_test=X_test.drop(columns=['name'])


# In[82]:


X_test


# In[83]:


X_train


# In[84]:


from sklearn.preprocessing import MinMaxScaler


# In[85]:


m=MinMaxScaler()


# In[86]:


sns.distplot(X_train['company'])


# In[87]:


m.fit(X_train)


# In[88]:


X_train=m.transform(X_train)


# In[89]:


X_test=m.transform(X_test)


# In[90]:


X_train


# In[91]:


X1=pd.DataFrame(X_train)


# In[92]:


X1


# In[93]:


X1.rename(columns={0:'Company'},inplace=True)


# In[94]:


sns.distplot(X1['Company'])


# In[99]:


from sklearn.linear_model import LinearRegression


# In[101]:


from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# In[102]:


lin=LinearRegression()


# In[104]:


x=lin.fit(X_train,y_train)


# In[105]:


x


# In[106]:


y_pred=x.predict(X_test)


# In[110]:


r=r2_score(y_test,y_pred)


# In[115]:


import pickle


# In[116]:


path='E:\Courses\Introduction to Data Science'


# In[117]:


pickle.dump(x,open('LinearRegression.pkl','wb'))


# In[ ]:




