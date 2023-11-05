#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\Advertising.csv")


# In[3]:


df


# In[4]:


df.info()


# In[6]:


df.describe()


# In[12]:


df.drop(columns='Unnamed: 0',inplace=True)


# In[13]:


df


# In[14]:


from sklearn.model_selection import train_test_split


# In[18]:


X=df['TV']
Y=df['Sales']


# In[19]:


plt.scatter(X,Y)


# In[21]:


from pandas.plotting import scatter_matrix


# In[23]:


scatter_matrix(df)
plt.show()


# In[24]:


from sklearn.linear_model import LinearRegression


# In[25]:


from sklearn.metrics import mean_squared_error, r2_score


# In[26]:


X_train,X_test,Y_train,Y_test=train_test_split(df['TV'],df['Sales'],test_size=0.2,random_state=42)


# In[28]:


model=LinearRegression()


# In[31]:


X_train=pd.DataFrame(X_train)


# In[32]:


X_train


# In[33]:


Y_train=pd.DataFrame(Y_train)


# In[34]:


Y_train


# In[35]:


model.fit(X_train,Y_train)


# In[37]:


pred=model.predict(pd.DataFrame(X_test))


# In[38]:


mse=mean_squared_error(Y_test,pred)


# In[39]:


r2=r2_score(Y_test,pred)


# In[40]:


mse


# In[41]:


r2


# In[44]:


plt.scatter(X_test,Y_test,label='TEST DATA',c='r',marker='x')
plt.plot(X_test,pred,c='b',linewidth=2,label='Regression Line')
plt.xlabel('TV')
plt.ylabel('SALES')
plt.legend(bbox_to_anchor=(1,1))
plt.show()


# In[46]:


import scipy.stats as stats


# In[53]:


residuals=pd.DataFrame(df['Sales'])-model.predict(pd.DataFrame(df['TV']))


# In[54]:


residuals


# In[60]:


X_coef=model.coef_[0][0]


# In[61]:


X_std_error=np.std(residuals)/np.std(df['TV'])


# In[62]:


X_std_error


# In[64]:


t_stats=X_coef/X_std_error


# In[65]:


t_stats


# In[66]:


dof=len(df['TV'])-2


# In[67]:


p=2*(1-stats.t.cdf(np.abs(t_stats),df))


# In[68]:


p


# In[72]:


if len(p>0.05) == 200:
    print('NOT STATISTICALLY SIGNIFICANT')
else:
    print('STATISTICALLY SIGNIFICANT')
    


# In[ ]:




