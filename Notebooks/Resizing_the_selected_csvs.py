#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


#Reading the Multi_category csv file
df=pd.read_csv('D:/Subjects/Intro To Data Science/Project/Multi_Category/2019-Oct.csv')
df_download=df.sample(frac=0.001).reset_index(drop=True)    #Sampling 1% of the data


# In[3]:


df_download.to_csv('Multi_Category_Store_1.csv',index=False) #Saving the new sampled data as a csv file


# In[4]:


#Reading the Electronics csv file
electronics=pd.read_csv('D:/Subjects/Intro To Data Science/Project/Electronics/events.csv')


# In[5]:


electronics1=electronics.sample(frac=0.005).reset_index(drop=True)    #Sampling 10% of the data
electronics1.to_csv('Electronics1.csv',index=False)                 #Saving the new sampled data as a csv file


# In[6]:


#Reading the Electronics 1 csv file
electronics1=pd.read_csv('D:/Subjects/Intro To Data Science/Project/Electronics1/events.csv')


# In[7]:


electronics1=electronics1.sample(frac=0.01).reset_index(drop=True)    #Sampling 10% of the data
electronics1.to_csv('Electronics2.csv',index=False)     #Saving the new sampled data as a csv file


# In[ ]:





# In[ ]:





# In[ ]:




