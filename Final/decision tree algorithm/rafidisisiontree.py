#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn import tree


# In[24]:


df = pd.read_csv("C:/Users/user/Desktop/rafi1.csv")


# In[25]:


df


# In[26]:


x = df.iloc[:,:-1]


# In[27]:


x


# In[28]:


y=df.iloc[:,3]


# In[29]:


y


# In[35]:


classify_ = tree.DecisionTreeClassifier()


# In[38]:


classify_ =classify_.fit(x,y)


# In[40]:


prediction_ = classify_.predict([[190,70,43]])


# In[41]:


prediction_


# In[ ]:




