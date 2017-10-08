
# coding: utf-8

# In[7]:


from sklearn import datasets,metrics
iris = datasets.load_iris()


# In[8]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(iris.data,iris.target,test_size=0.2,random_state=42)


# In[9]:


from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)


# In[10]:


neigh.fit(xtrain, ytrain)


# In[11]:


predi_test = neigh.predict(xtest)


# In[12]:


accuracy = metrics.accuracy_score(ytest,predi_test)


# In[13]:


print(accuracy)

