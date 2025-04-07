#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")


# In[27]:


df=pd.read_csv('merged_df.csv')


# In[28]:


df.head()


# In[29]:


#displaying statistics for numerical columns
df.describe()


# In[30]:


#checking for skewness in the datafram
df.skew(numeric_only=True)


# In[31]:


#checking for price distribution
plt.figure(figsize=(8,5))
ax=sns.histplot(df['price_scaled'], bins=60, kde=True,color='blue')
ax.set_xlim(0,1)
plt.title('Distribution of Scaled Price')
plt.xlabel('Scaled Price')
plt.ylabel('Count')
plt.show()


# In[32]:


#plotting box plot to detect outliers in the column 'Price'
plt.figure(figsize=(8,5))
ax=sns.boxplot(x=df['price_scaled'],native_scale=True, color='skyblue', width=0.5, 
               flierprops={'marker':'o','markersize':5, 'markerfacecolor':'red'})
ax.set_xlim(0, 0.5)
plt.xlabel('Scaled Price', fontsize=12)
plt.title('Box plot of scaled price')
plt.show()


# In[33]:


#Plotting the distribution of event_types
eventtype=df['event_type'].value_counts().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=eventtype, x=eventtype['event_type'], y=eventtype['count'],color='steelblue')
for index, value in enumerate(eventtype['count']):
    plt.text(index, value+1000, str(value), ha='center', fontsize=10)
plt.xlabel('Event Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of event_type values')
plt.show()


# In[34]:


#finding the top 10 brands whose products are bought by users
brand=df['brand'].value_counts()[0:10].reset_index()
plt.figure(figsize=(10,5))
sns.barplot(data=brand, x=brand['count'], y=brand['brand'],color='turquoise')

for index, value in enumerate(brand['count']):
    plt.text(value, index, str(value), va='center', fontsize=8)
plt.grid(axis='x', linestyle='--')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Brand', fontsize=12)
plt.title('Top 10 brands by number of products sold')
plt.show()


# In[45]:


#finding the top 10 products bought by users
products=df['category_code'].value_counts()[:10].reset_index()
products['short_category_code']=products['category_code'].apply(lambda x: x.split('.')[-1] if isinstance(x,str) else x)
plt.figure(figsize=(8,5))
sns.barplot(data=products,x=products['count'],y=products['short_category_code'], color='turquoise')
for index, value in enumerate(products['count']):
    plt.text(value,index,str(value), va='center', fontsize=8)
plt.grid(axis='x', linestyle='--')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Products', fontsize=12)
plt.title('Top 10 products bought by users')
plt.show()


# In[ ]:




