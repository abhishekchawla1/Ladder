#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve,roc_auc_score,auc
df=pd.read_csv(r"C:\Users\ASUS\Downloads\train.csv")
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
df['engagement']=l.fit_transform(df['engagement'])
df_test=pd.read_csv(r'C:\Users\ASUS\Downloads\test.csv')
df=df[['document_entropy', 'freshness', 'easiness','fraction_stopword_presence', 'speaker_speed', 'silent_period_rate','engagement']]
df_test_2=df_test[['document_entropy', 'freshness', 'easiness','fraction_stopword_presence', 'speaker_speed', 'silent_period_rate']]
X=df.drop(columns='engagement')
y=df['engagement']
gb=GradientBoostingClassifier()
gb.fit(X,y)
y_pred=gb.predict_proba(df_test_2)
rec=pd.Series(y_pred[:,1],index=df_test['id'],name='engagement',dtype='float32')

