#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports for array-handling and plotting
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# keep our keras backend tensorflow quiet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


if 0:
    get_ipython().system('pip install mlflow')
import mlflow
mlflow.tracking.set_tracking_uri(r'http://127.0.0.1:5000')
print('start tracking on: ',mlflow.tracking.get_tracking_uri()) 
import datetime 
exp_id = mlflow.set_experiment('exp-'+ '15') #str(datetime.datetime.now().second))
print(f'start experiment {exp_id}')


# In[3]:


def build_model():
    # building a linear stack of layers with the sequential model
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    print(model.summary())

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model


# In[4]:


def plot_digits(X_train, y_train):
    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(X_train[i], cmap='gray', interpolation='none')
        plt.title("Class {}".format(y_train[i]))
        plt.xticks([])
        plt.yticks([])
    fig.savefig(r'./output/digits.png', bbox_inches='tight')
    mlflow.log_artifact(r'output/digits.png')
    fig

def plot_results(model):
    # plotting the metrics
    fig = plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2,1,2)
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    fig.savefig(r'output/results.png', bbox_inches='tight')
    mlflow.log_artifact(r'output/results.png')
    fig


# In[5]:


def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return np.max(exps / np.sum(exps))


# ### load mnist train & test

# In[6]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# print the shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)


# ### building the input vector from the 28x28 pixels  

# In[7]:


f = plt.figure()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)     
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)


# ### one-hot encoding using keras' numpy-related utilities

# In[8]:


n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)


# ### split train to labeled & unlabeld

# In[9]:


labeled_size = .02
mlflow.log_param("labeled_size", labeled_size)

X_labeled, X_unlabeled, Y_labeled, Y_unlabeled = train_test_split(X_train, Y_train,
                                                                      train_size=labeled_size, stratify=y_train)
print("X_labeled shape", X_labeled.shape)
print("Y_labeled shape", Y_labeled.shape)
print("X_unlabeled shape", X_unlabeled.shape)
print("Y_unlabeled shape", Y_unlabeled.shape)
mlflow.log_param("labeled_sampels", X_labeled.shape[0])
mlflow.log_param("unlabeled_sampels", X_unlabeled.shape[0])


# ### build model :

# In[10]:


model = build_model() 


# #### loop few iterations 

# #### training the model on the labeled data and saving metrics in history

# In[11]:


iteration = 0
history = model.fit(X_labeled, Y_labeled, batch_size=128, epochs=10, verbose=4, validation_data=(X_test, Y_test))

# check accuracy on test data
metrics = model.evaluate(X_test, y=Y_test, batch_size=128)

print('\n accuracy after {} iterations :'.format(iteration))

for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))
    mlflow.log_metric(str(model.metrics_names[i]), metrics[i])   


# In[12]:


plot_results(history)


# In[13]:


plot_digits(X_test.reshape(X_test.shape[0],28,28),y_test)


# ### use the model to predict unlabeled data and create pseudo labeles

# In[14]:


iteration = iteration+1

threshold = .98
predict_proba = model.predict_proba(X_unlabeled)

# select unlabeled samples with predictions probabilty above threshold :
# TODO :softmax probas 
pseudo_index = np.max(predict_proba,axis=1) > threshold 
mlflow.log_param('threshold',threshold)   
print (f'\n using threshold of {threshold} - predicts {pseudo_index[pseudo_index].shape} pseudo labels from {pseudo_index.shape} unlabeld samples')
mlflow.log_param("pseudo labels", pseudo_index[pseudo_index].shape)

predict = model.predict(X_unlabeled)
X_pseudo_labeled = X_unlabeled[pseudo_index]
Y_pseudo_labeled = predict[pseudo_index]                           

# add pseudo labeled samples to labeled data :
X_mixed_labeled = np.concatenate((X_labeled, X_pseudo_labeled ), axis=0)
Y_mixed_labeled = np.concatenate((Y_labeled, Y_pseudo_labeled ), axis=0)


# ### check pseudo labels (compare to original train labels) : 

# In[15]:


y_pseudo_labeled = np.argmax(Y_pseudo_labeled,axis=1)
y_unlabeled = np.argmax(Y_unlabeled,axis=1)


# In[16]:


plot_digits(X_pseudo_labeled.reshape(X_pseudo_labeled.shape[0],28,28),y_pseudo_labeled)


#    ### training the model on the pseudo labeled data and saving metrics in history

# In[17]:


history = model.fit(X_mixed_labeled, Y_mixed_labeled, batch_size=128, epochs=10, verbose=2, validation_data=(X_test, Y_test))

# check accuracy on test data
metrics = model.evaluate(X_test, y=Y_test, batch_size=128)
print('\n accuracy after {} iterations :'.format(iteration))
for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))
    mlflow.log_metric('mixed_labeled_'+ str(model.metrics_names[i]), metrics[i])     
    print('\n')


# In[18]:


plot_results(history)
figsize=(15,15)


# #### train on new random initialized model 

# In[19]:


new_model = build_model()


# In[20]:


history = new_model.fit(X_mixed_labeled, Y_mixed_labeled, batch_size=128, epochs=10, verbose=2, validation_data=(X_test, Y_test))

# check accuracy on test data
metrics = new_model.evaluate(X_test, y=Y_test, batch_size=128)
print('\n accuracy on new model with pseudo labels :')
for i in range(len(model.metrics_names)):
    print(str(model.metrics_names[i]) + ": " + str(metrics[i]))    
    print('\n')
    mlflow.log_metric('mixed_new_model_'+str(model.metrics_names[i]), metrics[i]) 


# In[21]:


plot_results(history)


# In[ ]:




