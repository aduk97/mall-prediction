#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score






# In[5]:


data = pd.read_csv('C:\\Users\\adars\\OneDrive\\Desktop\\CERTIFICATIONS\\Mall_Customers.csv')
data.head()
data.tail()
data.shape
data.columns
data.isnull().sum()


# In[6]:


data.drop(["CustomerID"], axis = 1, inplace=True)
plt.figure(figsize=(10,6))
plt.title("Ages Frequency")
sns.axes_style("dark")
sns.violinplot(y=data["Age"])
plt.show()


# In[7]:


data.describe()
data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),cbar = True, square = True, fmt='.1f',annot=True,cmap='Greens')
plt.figure(1, figsize=(16,10))
sns.pairplot(data, hue='Gender')
plt.show()


# In[8]:


data['Gender'] = data['Gender'].map({'Male': 1, "Female": 0})
data
plt.figure(1, figsize=(16,4))
n = 0 
for x in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
    n = n + 1
    plt.subplot(1, 3, n)
    plt.subplots_adjust(hspace=0.5 , wspace=0.5)
    sns.distplot(data[x] , bins=10)
    plt.title('Distplot of {}'.format(x))
plt.show()


# In[9]:


X = data.iloc[:, -4:]
X
km_inertias, km_scores = [], []
for k in range(2, 9):
    km = KMeans(n_clusters=k).fit(X)
    km_inertias.append(km.inertia_)
    km_scores.append(silhouette_score(X, km.labels_))
    print(f"Processing K-Means with k = {k}, Intertia = {km.inertia_}, Silhoutte Score = {silhouette_score(X, km.labels_)}")
sns.lineplot(range(2,9), km_inertias)
plt.title('elbow graph / inertia depending on k')
plt.show()
sns.lineplot(range(2,9), km_scores)
plt.title('scores depending on k')
plt.show()


# In[10]:


km = KMeans(n_clusters=6).fit(X)
X
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12,8))
ax = Axes3D(fig)
x_age = X['Age']
x_income = X['Annual Income (k$)']
x_score = X['Spending Score (1-100)']
im = ax.scatter(x_age,x_income,x_score, s=50, alpha=0.6, c=km.labels_, cmap='RdYlGn')
fig.colorbar(im, ax=ax)
ax.set_xlabel('Age')
ax.set_ylabel('Annual income')
ax.set_zlabel('Spending score')
plt.show()


# In[ ]:




