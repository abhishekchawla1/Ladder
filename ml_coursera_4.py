#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\train.csv")


# In[3]:


df


# In[4]:


df_test=pd.read_csv(r"C:\Users\ASUS\Downloads\test.csv")


# In[5]:


df_test


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().any()


# In[9]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df['engagement']=l.fit_transform(df['engagement'])


# In[10]:


X=df.iloc[:,0:9]


# In[11]:


X.shape


# In[12]:


X


# In[13]:


y=df.iloc[:,-1]


# In[14]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[15]:


from sklearn.model_selection import GridSearchCV


# In[16]:


sc=StandardScaler()


# In[17]:


X=sc.fit_transform(X)


# In[18]:


X


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[20]:


from sklearn.svm import SVC


# In[21]:


from sklearn.tree import DecisionTreeClassifier


# In[22]:


from sklearn.ensemble import RandomForestClassifier


# In[23]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


# In[24]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[25]:


from xgboost import XGBClassifier


# In[26]:


from sklearn.metrics import roc_auc_score,auc,roc_curve,confusion_matrix


# In[27]:


x=['LogisticRegression','SVC','DecisionTreeClassifier','RandomForestClassifier','KNeighborsClassifier','BernoulliNB','GaussianNB','AdaBoostClassifier','GradientBoostingClassifier','XGBClassifier']


# In[28]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)
for i in x:
    model= eval(i)() if i != 'XGBClassifier' else XGBClassifier()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    roc=roc_auc_score(y_test,y_pred)
    print(f'ROC_AUC score of {i} = {roc}')
    


# In[29]:


def GSCV(model,param_grid):
    g=GridSearchCV(model, param_grid=param_grid,cv=3,verbose=2,n_jobs=-1,scoring='roc_auc')
    g.fit(X_train,y_train)
    return g.best_params_


# In[30]:


for i in x:
    model=eval(i)() if i!=XGBClassifier else XGBClassifier()
    if isinstance(model,LogisticRegression):
        param={'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1]}
    if isinstance(model,SVC):
        param={}
    if isinstance(model,DecisionTreeClassifier):
        param={'max_depth':[1,5,10,15],'criterion':['gini','entropy']}
    if isinstance(model,RandomForestClassifier):
        param={'n_estimators':[1,3,10,100,120],'max_depth':[1,5,10,15],'max_samples':[0.25,0.5,0.75,1],'bootstrap':[True,False],'max_features':[0.25,0.5,0.75,1]}
    if isinstance(model,KNeighborsClassifier):
        param={'n_neighbors':[1,3,5,7,10]}
    if isinstance(model,BernoulliNB):
        param={'alpha':[0.2,0.5,0.7,1]}
    if isinstance(model,AdaBoostClassifier):
        param={'n_estimators':[1,10,100,120],'learning_rate':[0.001,0.01,0.1,1],'algorithmm':['SAMME','SAMME.R']}
    if isinstance(model,GradientBoostingClassifier):
        param={'n_estimators':[0,10,100,120],'learning_rate':[0.001,0.01,0.1,1]}
    if isinstance(model,XGBClassifier):
        param={'n_estimators':[1,10,100,120,150,200]}
    else:
        param={}
                  
    best_param=GSCV(model,param)
    New_model=eval(i)(**best_param) if i!=XGBClassifier else XGBClassifier(**best_param)
    New_model.fit(X_train,y_train)
    y_pred_new=New_model.predict(X_test)
    roc=roc_auc_score(y_test,y_pred)
    print(f'ROC AUC SCORE OF {i} = {roc}')
    


# In[31]:


df


# In[36]:


def engage_model(X,y,df_test):
    s=StandardScaler()
    X=s.fit_transform(X)
    model=XGBClassifier()
    model.fit(X,y)
    df_test=s.transform(df_test)
    y_pred=model.predict_proba(df_test)
    return pd.Series(y_pred[:,1],index=np.arange(len(df_test))+1,dtype='float32')


# In[37]:


s=engage_model(X,y,df_test)


# In[38]:


s


# In[39]:


plt.plot(s)


# In[44]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True)


# In[45]:


df


# In[48]:


df_x=df[['document_entropy','freshness','easiness','fraction_stopword_presence','speaker_speed','silent_period_rate']]


# In[51]:


df_y=y


# In[52]:


df_x


# In[53]:


y


# In[54]:


from sklearn.preprocessing import MinMaxScaler


# In[55]:


m=MinMaxScaler()


# In[59]:


df_x=m.fit_transform(df_x)


# In[60]:


df_x


# In[62]:


from xgboost import XGBClassifier
df_x_train,df_x_test,df_y_train,df_y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=42)
x=XGBClassifier()
x.fit(df_x_train,df_y_train)
y_pred=x.predict(df_x_test)
r=roc_auc_score(df_y_test,y_pred)
r


# In[65]:


from sklearn.neural_network import MLPClassifier


# In[66]:


nn=MLPClassifier()


# In[67]:


pg={'hidden_layer_sizes':[100,200,(100,100),(100,100,100),(500,1000)],'activation':['identity', 'logistic', 'tanh', 'relu'],'solver':['lbfgs', 'sgd', 'adam']}
grid=GridSearchCV(nn,param_grid=pg,n_jobs=-1,verbose=2,cv=5)


# In[68]:


grid.fit(X_train,y_train)


# In[69]:


grid.best_score_


# In[71]:


grid.best_params_

