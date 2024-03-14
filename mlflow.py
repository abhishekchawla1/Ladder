#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlflow')


# In[2]:


import mlflow
import mlflow.sklearn


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


get_ipython().system('conda create -n mlflowtest python=3.8 -y')


# In[5]:


get_ipython().system('conda activate mlflowtest')


# In[6]:


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[7]:


from mlflow.models import infer_signature


# In[8]:


data=load_iris()


# In[9]:


df=pd.DataFrame(data.data,columns=data.feature_names)


# In[10]:


df


# In[11]:


df['label']=data.target


# In[12]:


df


# In[13]:


import matplotlib.pyplot as plt
plt.scatter(df['sepal length (cm)'],df['sepal width (cm)'],c=df['label'])


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(df.drop(columns='label'),df['label'],random_state=42,test_size=0.2)


# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


from sklearn.model_selection import GridSearchCV


# In[17]:


l=LogisticRegression()
params={'solver':['liblinear','lbfgs','sag','saga','newton-cg'],'max_iter':[100,1000,1500],'C':[0.0001,0.001,0.01,0.1,0.25,0.5,0.75,1]}
g=GridSearchCV(l,param_grid=params,cv=5,verbose=1,scoring='accuracy')


# In[18]:


g


# In[19]:


g.fit(X_train,y_train)


# In[20]:


g.best_params_


# In[21]:


g.best_score_


# In[22]:


log_reg=LogisticRegression(**g.best_params_)


# In[23]:


log_reg


# In[24]:


log_reg.fit(X_train,y_train)
y_pred=log_reg.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
accuracy


# In[25]:


from mlxtend.plotting import plot_decision_regions
import numpy as np


# In[26]:


from sklearn.decomposition import PCA


# In[27]:


p=PCA(n_components=2)


# In[28]:


X_train_pca=p.fit_transform(X_train)
X_test_pca=p.transform(X_test)


# In[29]:


lo=LogisticRegression(**g.best_params_)


# In[30]:


lo.fit(X_train_pca,y_train)


# In[31]:


y_pred_pca=lo.predict(X_test_pca)


# In[32]:


accuracy_score(y_test,y_pred_pca)


# In[33]:


plot_decision_regions(X_test_pca,np.array(y_test),lo)


# In[34]:


plot_decision_regions(X_train_pca,np.array(y_train),lo)


# In[35]:


mlflow.set_experiment('Exp1')
with mlflow.start_run():
    mlflow.log_params(g.best_params_)
    mlflow.log_metric('accuracy',accuracy)
    mlflow.set_tag('Information','Logistic Regression model for Iris Dataset')
    sig=infer_signature(X_train,log_reg.predict(X_train))
    info=mlflow.sklearn.log_model(sk_model=log_reg,artifact_path='Iris Model',signature=sig,input_example=X_train,registered_model_name='Iris Quickstart')    


# In[36]:


model=mlflow.pyfunc.load_model(info.model_uri)


# In[37]:


model


# In[38]:


preds=model.predict(X_test)


# In[39]:


data=X_test


# In[40]:


data['Actual Label']=y_test
data['Predicted Label']=preds


# In[41]:


data


# In[ ]:





# In[ ]:




