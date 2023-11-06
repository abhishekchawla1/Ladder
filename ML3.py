#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\insurance.csv")


# In[3]:


df


# In[4]:


df.isnull().sum()


# In[5]:


df['region'].unique()


# In[6]:


df.duplicated().any()


# In[7]:


df.drop_duplicates(inplace=True)


# In[8]:


df.shape


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


rows,columns=df.shape


# In[12]:


rows


# In[13]:


df.info()


# In[14]:


columns


# In[15]:


df.columns


# In[16]:


df.dtypes


# In[17]:


df.describe()


# In[18]:


df.head()


# In[19]:


pd.get_dummies(df['children'])


# In[20]:


df


# In[21]:


df['sex'].value_counts()


# In[22]:


sns.countplot(data=df,x='sex')


# In[23]:


df['sex']=df['sex'].map({'female':0,'male':1})


# In[24]:


df['smoker']=df['smoker'].map({'yes':1,'no':0})


# In[25]:


df['region']=df['region'].map({'southwest':4,'northwest':3,'southeast':2,'northeast':1})


# In[26]:


df


# In[27]:


X=df.drop(['charges'],axis=1)


# In[28]:


y=df['charges']


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[31]:


X_train


# In[32]:


from sklearn.linear_model import LinearRegression


# In[33]:


from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[34]:


lr=LinearRegression()


# In[35]:


lr.fit(X_train,y_train)


# In[36]:


svm=SVR()


# In[37]:


svm.fit(X_train,y_train)


# In[38]:


rf=RandomForestRegressor()


# In[39]:


rf.fit(X_train,y_train)


# In[40]:


gbr=GradientBoostingRegressor()


# In[41]:


gbr.fit(X_train,y_train)


# In[42]:


pred1=lr.predict(X_test)
pred2=svm.predict(X_test)
pred3=rf.predict(X_test)
pred4=gbr.predict(X_test)


# In[43]:


values=pd.DataFrame({'Actual':y_test,'Linear Regression':pred1,'Support Vector Regression':pred2,'Random Forest Regression':pred3,'Gradient Boosting Regression':pred4})


# In[44]:


values


# In[45]:


plt.subplot(221)
plt.plot(values['Actual'].iloc[0:30],label='Actual')
plt.plot(values['Linear Regression'].iloc[0:30],label='Linear Regression')


plt.subplot(222)
plt.plot(values['Actual'].iloc[0:30],label='Actual')
plt.plot(values['Support Vector Regression'].iloc[0:30],label='Support Vector Regression')


plt.subplot(223)
plt.plot(values['Actual'].iloc[0:30],label='Actual')
plt.plot(values['Random Forest Regression'].iloc[0:30],label='Rndom Forest Regressor')


plt.subplot(224)
plt.plot(values['Actual'].iloc[0:30],label='Actual')
plt.plot(values['Gradient Boosting Regression'].iloc[0:30],label='Gradient Boosting Regressor')

plt.tight_layout()
plt.show()


# In[46]:


pd.plotting.scatter_matrix(values)
plt.tight_layout()
plt.show()


# In[47]:


from sklearn.metrics import r2_score


# In[48]:


score1=r2_score(y_test,pred1)
score2=r2_score(y_test,pred2)
score3=r2_score(y_test,pred3)
score4=r2_score(y_test,pred4)


# In[49]:


scores=[score1,score2,score3,score4]


# In[50]:


scores


# In[51]:


from sklearn.metrics import mean_absolute_error


# In[52]:


s1=mean_absolute_error(y_test,pred1)
s2=mean_absolute_error(y_test,pred2)
s3=mean_absolute_error(y_test,pred3)
s4=mean_absolute_error(y_test,pred4)


# In[53]:


s=[s1,s2,s3,s4]


# In[54]:


s


# In[55]:


data={'age':40,'sex':1,'bmi':40.30,'children':4,'smoker':1,'region':2}


# In[56]:


data=pd.DataFrame(data,index=[1])


# In[57]:


data


# In[58]:


gbr.predict(data)


# In[59]:


g=GradientBoostingRegressor()


# In[60]:


X


# In[61]:


y


# In[62]:


g.fit(X,y)


# In[63]:


g.predict(data)


# In[5]:


import joblib


# In[65]:


joblib.dump(g,'Health_Insurance_Cost')


# In[66]:


model=joblib.load('Health_Insurance_Cost')


# In[67]:


model.predict(data)


# In[1]:


from tkinter import *


# In[8]:


def show_entry():
    
    p1=float(e1.get())
    p2=float(e2.get())
    p3=float(e3.get())
    p4=float(e4.get())
    p5=float(e5.get())
    p6=float(e6.get())
    
    model=joblib.load('Health_Insurance_Cost')
    result=model.predict([[p1,p2,p3,p4,p5,p6]])
    
    Label(master,text='Insurance Cost').grid(row=7)
    Label(master,text=result).grid(row=8)


# In[9]:


master=Tk()
master.title('Insurance Cost Prediction')
label=Label(master,text='Insurance Cost Prediction',bg='black',fg='white').grid(row=0,columnspan=2)
Label(master,text='Enter your age').grid(row=1)
Label(master,text='Male or Female [0-1]').grid(row=2)
Label(master,text='Enter your BMI value').grid(row=3)
Label(master,text='Enter Number of children').grid(row=4)
Label(master,text='Smoker Yes=1, No=0').grid(row=5)
Label(master,text='Region [1-4]').grid(row=6)

e1=Entry(master)
e2=Entry(master)
e3=Entry(master)
e4=Entry(master)
e5=Entry(master)
e6=Entry(master)

e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)

Button(master,text='Predict',command=show_entry).grid()

mainloop()


# In[ ]:




