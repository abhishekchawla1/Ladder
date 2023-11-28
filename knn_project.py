#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[3]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\networkadds.csv")


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.duplicated().any()


# In[9]:


X=df[['Age','EstimatedSalary']]


# In[12]:


y=df[['Purchased']]


# In[13]:


y


# In[14]:


y.shape


# In[15]:


X.shape


# In[16]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[21]:


X_train


# In[22]:


X_test


# In[23]:


y_train


# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


sc=StandardScaler()


# In[26]:


X_train=sc.fit_transform(X_train)


# In[27]:


X_train


# In[28]:


X_test=sc.fit_transform(X_test)


# In[29]:


X_test


# In[33]:


from sklearn.neighbors import KNeighborsClassifier


# In[37]:


k=np.sqrt(X_train.shape[0]).astype(int)
k


# In[38]:


knn=KNeighborsClassifier(n_neighbors=k)


# In[40]:


knn.fit(X_train,y_train)


# In[41]:


y_pred=knn.predict(X_test)


# In[42]:


from sklearn.metrics import accuracy_score


# In[43]:


y_pred.shape


# In[44]:


y_test.shape


# In[45]:


y_test=np.array(y_test)


# In[46]:


y_test.shape


# In[47]:


a=accuracy_score(y_test,y_pred)


# In[48]:


a


# In[49]:


from sklearn.metrics import confusion_matrix


# In[50]:


c=confusion_matrix(y_test,y_pred)


# In[51]:


c


# In[54]:


accuracy=[]
for i in range(1,26):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    accuracy.append(accuracy_score(y_test,y_pred))


# In[55]:


accuracy


# In[56]:


plt.plot(range(1,26),accuracy)


# In[64]:


def predict_output():
    age=int(input('ENTER THE AGE'))
    salary=int(input('ENTER SALARY'))
    X_new=np.array([[age],[salary]]).reshape(1,2)
    X_new=sc.transform(X_new)
    p=knn.predict(X_new)
    if p==1:
        return 'Will Purchase'
    else:
        return 'Will Not Purchase'


# In[65]:


predict_output()


# In[67]:


a=np.arange(X_train[:,0].min()-1,X_train[:,0].max()+1,step=0.01)
b=np.arange(X_train[:,1].min()-1,X_train[:,1].max()+1,step=0.01)


# In[68]:


xx,yy=np.meshgrid(a,b)


# In[69]:


xx.shape


# In[70]:


yy.shape


# In[73]:


i=np.array([xx.ravel(),yy.ravel()]).T


# In[74]:


i


# In[76]:


knn=KNeighborsClassifier(n_neighbors=11)


# In[77]:


knn.fit(X_train,y_train)


# In[78]:


l=knn.predict(i)


# In[86]:


from matplotlib import cm


# In[88]:


cmap=cm.get_cmap('cool')


# In[96]:


plt.contour(xx,yy,l.reshape(xx.shape),cmap=cmap)
plt.scatter(X_train[:,0],X_train[:,1],c=np.array(y_train))


# In[ ]:




