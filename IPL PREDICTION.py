#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


matches=pd.read_csv(r"C:\Users\ASUS\Downloads\matches.csv")
deli=pd.read_csv(r"C:\Users\ASUS\Downloads\deliveries.csv")


# In[3]:


matches.head()


# In[5]:


matches.shape


# In[6]:


matches.info()


# In[7]:


matches.isnull().sum()


# In[8]:


matches.duplicated().any()


# In[9]:


deli


# In[10]:


deli.shape


# In[11]:


df = deli.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
df = df[df['inning'] == 1]
df


# In[12]:


match_df = matches.merge(df[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[13]:


match_df


# In[14]:


match_df['team1'].unique()


# In[15]:


teams=['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
       'Kolkata Knight Riders', 'Kings XI Punjab',
       'Chennai Super Kings', 'Rajasthan Royals',
       'Delhi Capitals']


# In[16]:


match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')


# In[17]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[18]:


match_df.shape


# In[19]:


match_df = match_df[match_df['dl_applied'] == 0]
match_df = match_df[['match_id','city','winner','total_runs']]


# In[21]:


deli_df=match_df.merge(deli,on='match_id')


# In[23]:


deli_df=deli_df[deli_df['inning']==2]


# In[24]:


deli_df


# In[25]:


deli_df['current_score'] = deli_df.groupby('match_id').cumsum()['total_runs_y']


# In[26]:


deli_df['runs_left'] = deli_df['total_runs_x'] - deli_df['current_score']


# In[27]:


deli_df['balls_left'] = 126 - (deli_df['over']*6 + deli_df['ball'])


# In[28]:


deli_df


# In[29]:


deli_df['player_dismissed'] = deli_df['player_dismissed'].fillna("0")
deli_df['player_dismissed'] = deli_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
deli_df['player_dismissed'] = deli_df['player_dismissed'].astype('int')
wickets = deli_df.groupby('match_id').cumsum()['player_dismissed'].values
deli_df['wickets'] = 10 - wickets


# In[30]:


deli_df.head(2)


# In[32]:


deli_df['crr'] = (deli_df['current_score']*6)/(120 - deli_df['balls_left'])


# In[33]:


deli_df['rrr'] = (deli_df['runs_left']*6)/deli_df['balls_left']


# In[34]:


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0
deli_df['result'] = deli_df.apply(result,axis=1)


# In[35]:


final_df = deli_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]
final_df = final_df.sample(final_df.shape[0])
final_df.sample()


# In[36]:


final_df.dropna(inplace=True)


# In[37]:


final_df = final_df[final_df['balls_left'] != 0]


# In[38]:


X = final_df.iloc[:,:-1]


# In[39]:


y = final_df.iloc[:,-1]


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[41]:


X_train


# In[42]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')


# In[44]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# In[ ]:




