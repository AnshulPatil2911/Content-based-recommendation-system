#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")


# In[2]:


#Reading the resampled CSVs
multi=pd.read_csv('Multi_Category_Store_1.csv')
electronics=pd.read_csv('Electronics1.csv')
electronics1=pd.read_csv('Electronics2.csv')


# In[3]:


dfs=[multi,electronics,electronics1]         #Loading the CSVs into a list
dfs_name=['multi', 'electronics', 'electronics1']  #Storing the names of the CSVs in a list


# In[4]:


multi.head(5)


# In[5]:


electronics.head()


# In[6]:


electronics1.head()


# In[7]:


#Printing the lengths of the dataframes
def len_dfs(df, name):
    print(f'The shape of the dataframe {name} is {df.shape}')

for df, name in zip(dfs,dfs_name):
    len_dfs(df,name)


# In[8]:


#Printing the names of the columns
def column_names(df,name):
    print(f'The names of the columns in the dataframe {name} are: {df.columns.to_list()}')

for df, name in zip(dfs,dfs_name):
    column_names(df,name)


# In[9]:


#Printing the number of null values in each column
def columns_null_value(df, name):
    null_counts=df.isnull().sum()
    cols_with_null_values=null_counts[null_counts>0]
    print(f"{name} has missing values in the following columns:")
    print(cols_with_null_values.to_string())
    print(f'Length of the dataframe {name} is {len(df)}')
    print("-" * 40) 

for df, name in zip(dfs,dfs_name):
    columns_null_value(df,name)


# In[10]:


#Dropping the rows with null value from the 'user_session' columns
for df in dfs:
    df.dropna(subset=['user_session'],inplace=True)


# In[11]:


#Printing the rows with with missing values
def rows_with_missing_values(df, name):
    df_missing=df[df.isnull().any(axis=1)]
    print(df_missing.head())

for df, name in zip(dfs,dfs_name):
    rows_with_missing_values(df,name)


# In[12]:


#checking if there are some rows with missing values in 'category_code' or 'brand' have the same 'category_id' as rows without missing values.
def identifying_common_ids(df, name):

    missing_rows = df[df['category_code'].isnull() | df['brand'].isnull()]
    
    non_missing_rows = df.dropna(subset=['category_code', 'brand'])
    
    matching_category_ids = missing_rows['category_id'].isin(non_missing_rows['category_id'])

    if matching_category_ids.any():
        print("Some rows with missing values in 'category_code' or 'brand' have the same 'category_id' as rows without missing values.")
    else:
        print("No rows with missing values in 'category_code' or 'brand' have a matching 'category_id' in rows without missing values.")

    for col in ['category_code', 'brand']:
        df[col] = df.groupby('category_id')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
    
    print(len(missing_rows),len(non_missing_rows),len(matching_category_ids),matching_category_ids.value_counts())


    return df
# for df, name in zip(dfs,dfs_name):
#     identifying_common_ids(df,name)
dfs_filled = {name:identifying_common_ids(df, name) for df, name in zip(dfs, dfs_name)}


# In[13]:


#checking if there are some rows with missing values in 'category_code' or 'brand' have the same 'product_id' as rows without missing values.
def identifying_common_ids_product(df, name):

    missing_rows = df[df['category_code'].isnull() | df['brand'].isnull()]
    
    non_missing_rows = df.dropna(subset=['category_code', 'brand'])
    
    matching_category_ids = missing_rows['product_id'].isin(non_missing_rows['product_id'])
    
    if matching_category_ids.any():
        print("Some rows with missing values in 'category_code' or 'brand' have the same 'product_id' as rows without missing values.")
    else:
        print("No rows with missing values in 'category_code' or 'brand' have a matching 'product_id' in rows without missing values.")

    for col in ['category_code', 'brand']:
        df[col] = df.groupby('category_id')[col].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
    
    print(len(matching_category_ids),matching_category_ids.value_counts())


for df, name in zip(dfs,dfs_name):
    identifying_common_ids_product(df,name)


# In[14]:


#dropping the null values
for name,df in dfs_filled.items():
    print(f'The number of null values in the columns of the dataframe {name} are:')
    print(df.isnull().sum())
    print('Dropping the null values\n')
    df.dropna(subset=['category_code','brand'],inplace=True)


# In[15]:


#merging the three dataframes into a single one
merged_df=pd.concat([dfs_filled[dfs_name[0]],dfs_filled[dfs_name[1]],dfs_filled[dfs_name[2]]],ignore_index=True)
print(f'The length of the merged dataframe is {len(merged_df)}')


# In[16]:


#checking if the merged dataframe has any null values
merged_df.isnull().any()


# In[17]:


#checking for duplicated values in the dataframe
print(merged_df.duplicated().value_counts())


# In[18]:


#dropping the duplicate values
merged_df=merged_df.drop_duplicates(keep='first')


# In[19]:


merged_df.to_csv('merged_df.csv',index=False)


# ## Detecting outliers using IQR
# 

# In[20]:


#finding the  numerical columns
num_cols=merged_df.select_dtypes(include=['number']).columns
print('Numerical Columns:', num_cols)


# In[21]:


# #checking which of the columns have a normal distribution and which ones have a skewed distribution
# for col in num_cols:
#     plt.figure(figsize=(6,4))
#     sns.histplot(merged_df[col],kde=True, bins=10)
#     plt.title(f'Distribution of {col}')
#     plt.show()


# In[22]:


#checking the distribution type in the columns using statistical testing
for col in num_cols:
    stat, p= stats.shapiro(merged_df[col].dropna())
    print(f'{col}:p-value={p}')
    if p>0.05:
        print(f'{col} is normally distributed')
    else:
        print(f'{col} is not normally distributed')


# In[23]:


#detecting outliers
def detect_outliers(df,col):
    q1=df[col].quantile(0.25)
    q3=df[col].quantile(0.75)
    iqr=q3-q1
    lower_bound=q1-1.5*iqr
    upper_bound=q3+1.5*iqr
    return df[(df[col]<lower_bound) | (df[col]>upper_bound)][col]

outliers=detect_outliers(merged_df,'price')

print(outliers)


# In[24]:


#handling the outliers using the MinMaxScaler method
scaler=MinMaxScaler(feature_range=(0,10))
merged_df['price_scaled']=scaler.fit_transform(merged_df[['price']])


# In[ ]:




