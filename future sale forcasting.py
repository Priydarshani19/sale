#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[3]:


df = pd.read_csv(r"C:\Users\priyd\Downloads\Sales.csv")
df.head()


# In[5]:


df.isnull().sum()


# In[80]:


import matplotlib.pyplot as plt
plt.scatter(df.Sales,df.TV)


# In[22]:


import matplotlib.pyplot as plt
plt.scatter(df['Sales'],df["Radio"])


# In[23]:


import matplotlib.pyplot as plt
plt.scatter(df['Sales'],df["Newspaper"])


# In[ ]:


#prediction model


# In[27]:


x=df.drop("Sales",axis="columns")
x.head()


# In[28]:


y=df["Sales"]
y.head()


# In[ ]:


#split model into train and test


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=30)


# In[33]:


model = LinearRegression()


# In[34]:


#train model
model.fit(x_train,y_train)


# In[35]:


#check accuracy scor of model


# In[36]:


model.score(x_test,y_test)


# In[39]:


x_test.head()


# In[41]:


y_test.head()


# In[ ]:


#predict 


# In[42]:


model.predict([[104.6,5.7,34.4]])


# In[ ]:





# In[ ]:




