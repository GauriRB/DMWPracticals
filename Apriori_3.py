#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import apyori as apriori



# In[4]:


d=pd.read_csv('CanteenDataSet.csv')


# In[5]:


d


# In[6]:


d.head()


# In[7]:


d.isnull().sum()


# In[12]:


d.columns=['Dishes']
     


# In[13]:


transactions=[]
items=d['Dishes'].values
print(items)



# In[15]:


for i in range(0,len(items)):
    transactions.append(items[i].split(','))


# In[17]:


print(transactions)


# In[22]:


from mlxtend.frequent_patterns import apriori, association_rules

from mlxtend.preprocessing import TransactionEncoder


# In[25]:


encoder=TransactionEncoder()
tran=encoder.fit(transactions).transform(transactions)
print(tran)


# In[26]:


encodedData=pd.DataFrame(data=tran,columns=encoder.columns_,dtype=int)


# In[27]:


print(encodedData)


# In[28]:


encodedData.head()


# In[31]:


support=apriori(encodedData,min_support=0.2,use_colnames=True)
support.sort_values(by='support',ascending=False)
confidence=association_rules(support,metric='confidence',min_threshold=0.3)
confidence.sort_values(by='confidence',ascending=False)

