
# coding: utf-8

# In[1]:


from sklearn import datasets, metrics


# In[2]:


iris = datasets.load_iris()
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(iris.data, test_size=0.8, random_state=42)


# In[3]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()


# In[4]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)


# In[5]:


dtc.fit(x_train, y_train)


# In[6]:


prediction = dtc.predict(x_test)


# In[7]:



accuracy = metrics.accuracy_score(y_test, prediction)


# In[8]:



print(accuracy)

