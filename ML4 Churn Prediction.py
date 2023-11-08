#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\ASUS\Downloads\Churn_Modelling.csv")


# In[3]:


df


# In[4]:


df.head(2)


# In[5]:


df.tail()


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().any()


# In[8]:


r,c=df.shape


# In[9]:


r


# In[10]:


c


# In[11]:


df.columns


# In[12]:


len(df.columns)


# In[13]:


df.info()


# In[14]:


df.describe()


# In[15]:


df.describe(include='all')


# In[16]:


df.dtypes


# In[17]:


df.head(2)


# In[18]:


df.drop(columns=['RowNumber','Surname','CustomerId'],inplace=True)


# In[19]:


df.head(2)


# In[20]:


df['Geography'].unique()


# In[21]:


df['Gender']=df['Gender'].map({'Female':0,'Male':1})


# In[22]:


df


# In[23]:


new=pd.get_dummies(df['Geography'],drop_first=True)


# In[24]:


df=pd.get_dummies(df,drop_first=True)


# In[25]:


df


# In[26]:


df['Exited'].value_counts()


# In[27]:


sns.countplot(data=df,x='Exited')


# In[28]:


X=df.drop(columns=['Exited'])


# In[29]:


X


# In[30]:


y=df['Exited']


# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2,stratify=y)


# In[33]:


X_train


# In[34]:


#feature scaling


# In[35]:


from sklearn.preprocessing import StandardScaler


# In[36]:


sc=StandardScaler()


# In[37]:


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[38]:


from sklearn.linear_model import LogisticRegression


# In[39]:


lr=LogisticRegression()


# In[40]:


lr.fit(X_train,y_train)


# In[41]:


pred1=lr.predict(X_test)


# In[42]:


from sklearn.metrics import accuracy_score


# In[43]:


accuracy_score(y_test,pred1)


# In[44]:


from sklearn.metrics import precision_score,recall_score,f1_score


# In[45]:


precision_score(y_test,pred1)


# In[46]:


recall_score(y_test,pred1)


# In[47]:


f1_score(y_test,pred1)


# In[48]:


#accuracy is not used for imbalanced dataset


# In[49]:


#handling imbalanced data: oversampling and undersampling


# In[50]:


#smote: synthetic data points and no duplicates


# In[51]:


df


# In[52]:


pip install imblearn


# In[53]:


from imblearn.over_sampling import SMOTE


# In[54]:


X_res,y_res=SMOTE().fit_resample(X,y)


# In[55]:


X_res


# In[56]:


y_res


# In[57]:


sns.countplot(x=y_res)


# In[58]:


from sklearn.model_selection import train_test_split


# In[59]:


X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.2,random_state=42)


# In[60]:


from sklearn.linear_model import LogisticRegression


# In[61]:


model=LogisticRegression()


# In[62]:


model.fit(X_train,y_train)


# In[63]:


pred2=model.predict(X_test)


# In[64]:


from sklearn.preprocessing import StandardScaler


# In[65]:


sc=StandardScaler()


# In[66]:


X_train=sc.fit_transform(X_train)


# In[67]:


X_test=sc.transform(X_test)


# In[68]:


precision_score(y_test,pred2)


# In[69]:


recall_score(y_test,pred2)


# In[70]:


#SVC


# In[71]:


from sklearn import svm


# In[72]:


svm=svm.SVC()


# In[73]:


X_train


# In[74]:


svm.fit(X_train,y_train)


# In[75]:


pred3=svm.predict(X_test)


# In[76]:


accuracy_score(y_test,pred3)


# In[77]:


from sklearn.neighbors import KNeighborsClassifier


# In[78]:


knn=KNeighborsClassifier()


# In[79]:


knn.fit(X_train,y_train)


# In[80]:


pred4=knn.predict(X_test)


# In[81]:


accuracy_score(y_test,pred4)


# In[82]:


precision_score(y_test,pred4)


# In[83]:


from sklearn.tree import DecisionTreeClassifier


# In[84]:


dt=DecisionTreeClassifier()


# In[85]:


dt.fit(X_train,y_train)


# In[86]:


pred5=dt.predict(X_test)


# In[87]:


accuracy_score(y_test,pred5)


# In[88]:


precision_score(y_test,pred5)


# In[89]:


from sklearn.ensemble import RandomForestClassifier


# In[90]:


rf=RandomForestClassifier()


# In[91]:


rf.fit(X_train,y_train)


# In[92]:


pred6=rf.predict(X_test)


# In[93]:


accuracy_score(y_test,pred6)


# In[94]:


precision_score(y_test,pred6)


# In[95]:


from sklearn.ensemble import GradientBoostingClassifier


# In[96]:


gb=GradientBoostingClassifier()


# In[97]:


gb.fit(X_train,y_train)


# In[98]:


pred7=gb.predict(X_test)


# In[99]:


accuracy_score(y_test,pred7)


# In[100]:


precision_score(y_test,pred7)


# In[101]:


final_data=pd.DataFrame({'Models':['Logistic Regression','Support Vector Classification','K Nearest Neighbors','Decision Trees','Random Forest','Gradient Boosting'],'ACC':[accuracy_score(y_test,pred2),accuracy_score(y_test,pred3),accuracy_score(y_test,pred4),accuracy_score(y_test,pred5),accuracy_score(y_test,pred6),accuracy_score(y_test,pred7)]})


# In[102]:


final_data


# In[103]:


plt.bar(final_data['Models'],height=final_data['ACC'])
plt.xticks(rotation=60)


# In[104]:


#using Random Forest


# In[105]:


import joblib


# In[106]:


from sklearn.ensemble import RandomForestClassifier


# In[107]:


rc=RandomForestClassifier()


# In[108]:


from sklearn.preprocessing import StandardScaler


# In[109]:


sc=StandardScaler()


# In[110]:


X_res=sc.fit_transform(X_res)


# In[111]:


rc.fit(X_res,y_res)


# In[112]:


joblib.dump(rc,'CHURN PREDICTION')


# In[113]:


model=joblib.load('CHURN PREDICTION')


# In[114]:


df


# In[115]:


model.predict([[619,42,2,0.00,1,1,1,10134.88,0,0,0]])


# In[116]:


from tkinter import *
from sklearn.preprocessing import StandardScaler


# In[117]:


def show_entry():
    p1=int(e1.get())
    p2=int(e2.get())
    p3=int(e3.get())
    p4=float(e4.get())
    p5=int(e5.get())
    p6=int(e6.get())
    p7=int(e7.get())
    p8=float(e8.get())
    p9=int(e9.get())
    
    if p9==1:
        Geography_Germany=1
        Geography_Spain=0
        Geography_France=0
        
    if p9==2:
        Geography_Germany=0
        Geography_Spain=1
        Geography_France=0
        
    if p9==3:
        Geography_Germany=0
        Geography_Spain=0
        Geography_France=1
        
    p10=int(e10.get())
    
    model=joblib.load('CHURN PREDICTION')
    result=model.predict(sc.transform([[p1,p2,p3,p4,p5,p6,p7,p8,Geography_Germany,Geography_Spain,p10]]))

master=Tk()
master.title('BANK CHURN PREDICTOR')
label=Label(master,text='BANK CHURN PREDCTION',bg='black',fg='white').grid(row=0,columnspan=2)
Label(master,text='ENTER CREDIT SCORE').grid(row=1)
Label(master,text='ENTER AGE').grid(row=2)
Label(master,text='ENTER TENURE').grid(row=3)
Label(master,text='ENTER BALANCE').grid(row=4)
Label(master,text='ENTER NUMBER OF PRODUCTS').grid(row=5)
Label(master,text='WHETHER DO YOU HAVE A CREDIT CARD').grid(row=6)
Label(master,text='ARE YOU A MEMBER').grid(row=7)
Label(master,text='ESTIMATED SALARY').grid(row=8)
Label(master,text='Geography').grid(row=9)
Label(master,text='ENTER GENDER').grid(row=10)

e1=Entry(master)
e2=Entry(master)
e3=Entry(master)
e4=Entry(master)
e5=Entry(master)
e6=Entry(master)
e7=Entry(master)
e8=Entry(master)
e9=Entry(master)
e10=Entry(master)

e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)
e7.grid(row=7,column=1)
e8.grid(row=8,column=1)
e9.grid(row=9,column=1)
e10.grid(row=10,column=1)

Button(master,text='PREDICT',command=show_entry).grid()
mainloop()


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# In[ ]:




