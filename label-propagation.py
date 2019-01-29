
# coding: utf-8

# In[1]:


import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
print(iris.feature_names) 
print(iris.data[:5])


# In[2]:


rng = np.random.RandomState(42)
random_unlabeled_points = rng.rand(len(iris.target)) < 0.8
labels = np.copy(iris.target)
print('supervised labels: ')
print(labels)
labels[random_unlabeled_points] = -1
print('semi supervised labels: ')
print(labels)


# In[3]:


from sklearn.semi_supervised import LabelPropagation
label_prop_model = LabelPropagation()


# In[4]:


label_prop_model.fit(iris.data, labels)


# In[5]:


label_prop_model.transduction_



# In[6]:


label_prop_model.score(X=iris.data ,y=iris.target)

