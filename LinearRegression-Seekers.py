#!/usr/bin/env python
# coding: utf-8

# In[188]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[189]:


# import the data from a file and process the missing feilds
data =pd.read_csv("ML_Course_HW_1.csv")
# Check if there are any missing values in the data and fill them with medians of the same column
missing = data.isna().sum()
data = data.fillna( data.median() )
cols = list(data.columns)
cols.remove("CUST_ID")
cols.remove("PAYMENTS")
X = data[cols].iloc[ :, :].values
y = data["PAYMENTS"]
#from sklearn.preprocessing import StandardScaler 
#X_std = StandardScaler().fit_transform(X)


# In[190]:


#Splitting data set into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# In[191]:


print(np.shape(X_train))
print(np.shape(y_train))
# print (y_train[0])


# In[192]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)
regressor.fit(X_train, y_train)


# In[193]:


y_pred = regressor.predict(X_test)
print(np.shape(y_pred))


# In[194]:


print(X_train[:,2][0])
y_train_pred = regressor.predict(X_train)
print(np.shape(y_train_pred))


# In[195]:


output = regressor.predict(X_train)
print(np.shape(output))
plt.scatter(X_train[:,2], y_train, color = 'red')
plt.scatter(X_train[:,2], regressor.predict(X_train), color = 'blue')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# In[196]:


cols = list(data.columns)
cols.remove("CUST_ID")
cols.remove("CREDIT_LIMIT")
X = data[cols].iloc[ :, :].values
y = data["CREDIT_LIMIT"]
#from sklearn.preprocessing import StandardScaler 
#X_std = StandardScaler().fit_transform(X)


# In[197]:


#Splitting data set into Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# In[198]:


print(np.shape(X_train))
print(np.shape(y_train))
# print (y_train[0])


# In[199]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)
regressor.fit(X_train, y_train)


# In[200]:


y_pred = regressor.predict(X_test)
print(np.shape(y_pred))


# In[201]:


print(X_train[:,2][0])
y_train_pred = regressor.predict(X_train)
print(np.shape(y_train_pred))


# In[202]:


output = regressor.predict(X_train)
print(np.shape(output))
plt.scatter(X_train[:,2], y_train, color = 'red')
plt.scatter(X_train[:,2], regressor.predict(X_train), color = 'blue')
plt.title('Simple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

