#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier


# In[2]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\train.csv")


# In[3]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df['engagement']=l.fit_transform(df['engagement'])


# In[4]:


df


# In[5]:


X=df.drop(columns=['id','title_word_count','engagement'])


# In[6]:


y=df['engagement']


# In[7]:


from sklearn.preprocessing import StandardScaler


# In[8]:


sc=StandardScaler()


# In[9]:


X=sc.fit_transform(X)


# In[10]:


nn=MLPClassifier()


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=4)


# In[13]:


model=nn.fit(X_train,y_train)


# In[24]:


y_pred=model.predict(X_test)


# In[25]:


from sklearn.metrics import roc_curve,roc_auc_score


# In[26]:


roc=roc_auc_score(y_test,y_pred)


# In[27]:


roc


# In[43]:


fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1].reshape(-1,1))
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

