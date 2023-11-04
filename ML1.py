#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 


# In[4]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\add.csv")


# In[5]:


df


# In[6]:


sns.scatterplot(df['x'])


# In[7]:


df.head()


# In[8]:


plt.scatter(df['x'],df['sum'])


# In[9]:


plt.scatter(df['y'],df['sum'])


# In[10]:


#independent variable (feature matrix)
X=df[['x','y']]


# In[11]:


Y=df['sum']


# In[12]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=42)


# In[13]:


X


# In[14]:


Y


# In[15]:


X_train


# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


model=LinearRegression()


# In[18]:


model.fit(X_train,Y_train)


# In[19]:


model.score(X_train,Y_train)


# In[20]:


model.score(X_test,Y_test)


# In[21]:


pred=model.predict(X_test)


# In[22]:


pred


# In[23]:


Y_test


# In[24]:


data=pd.DataFrame({'Actual':Y_test,'Prediction':pred})


# In[25]:


data


# In[26]:


model.predict([[10,20]])


# In[27]:


#model saving using joblib


# In[33]:


pip install joblib


# In[28]:


import joblib


# In[29]:


joblib.dump(model,"Addition")


# In[30]:


model=joblib.load('Addition')


# In[31]:


model.predict([[10,40]])


# In[32]:


label=Label(master,text="ADDITION OF TWO NUMBERS USING MACHINE LEARNING",bg='black',fg='white').grid(row=0,columnspan=2)


# In[33]:


import tkinter as tk
from tkinter import Entry, Button, Label
import joblib

master = tk.Tk()
master.title("Addition Prediction")

def show_entry_fields():
    p1 = float(e1.get())
    p2 = float(e2.get())
    
    model = joblib.load('ADDITION')
    result = model.predict([[p1, p2]])
    
    result_label.config(text=f'SUM IS: {result[0]}')

Label(master, text="ENTER FIRST NUMBER").grid(row=1)
Label(master, text="ENTER SECOND NUMBER").grid(row=2)
e1 = Entry(master)
e2 = Entry(master)
e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
result_label = Label(master, text="")
result_label.grid(row=5)

Button(master, text='PREDICT', command=show_entry_fields).grid(row=3, columnspan=2)

master.mainloop()


# In[ ]:




