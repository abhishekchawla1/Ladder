#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import make_classification


# In[2]:


X,y=make_classification(n_samples=100,n_features=2,n_informative=1,n_classes=2,n_redundant=0,n_clusters_per_class=1,random_state=42,class_sep=5,hypercube=False)


# In[3]:


import matplotlib.pyplot as plt
import numpy as np


# In[4]:


plt.scatter(X[:,0],X[:,1],c=y,cmap='summer',s=52)


# In[5]:


X


# In[6]:


y


# In[11]:


def perceptron(X,y):
    X=np.insert(X,0,1,axis=1)
    weights=np.ones(X.shape[1])
    learning_rate=0.1
    
    for i in range(1000):
        j=np.random.randint(0,100)
        y_hat=step(np.dot(X[j],weights))
        weights=weights+learning_rate*(y[j]-y_hat)*X[j]
    
    return weights[0],weights[1:]
    


# In[12]:


def step(x):
    if x>0:
        return 1
    else:
        return 0


# In[13]:


intercept,coefficients=perceptron(X,y)


# In[14]:


coefficients


# In[15]:


intercept


# In[18]:


m=-(coefficients[0]/coefficients[1])
c=-(intercept/coefficients[1])


# In[21]:


x_input=np.linspace(-3,3,100)
y_input=m*x_input+c


# In[24]:


plt.figure(figsize=(10,5))
plt.plot(x_input,y_input,color='red',linewidth=2)
plt.scatter(X[:,0],X[:,1],c=y,cmap='winter',s=50)
plt.show()


# In[ ]:




