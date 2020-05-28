#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import librosa
import soundfile as sf


# In[8]:


# Function to load the samples from a path to a prepared CNN Train data shape[n_samples, WINDOWSIZE, 1] 
# AND the y data

def train_test_CNN(train_path, test_path, start, end, window=5, stride=1):
    
    train, sr = librosa.load(train_path)
    test, sr = librosa.load(test_path)
    
    train = train[sr * start: sr * end]
    test = test[sr * start: sr * end]
    original = train

    indexer = np.arange(window)[None, :] + stride*np.arange(len(test) + stride - window)[:, None]

    train = train[indexer]

    train = np.reshape(train, (train.shape[0], window, -1)) 
    
    test = test[: - (window -1)]
        
    return train, test, original
    


# In[9]:


from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D


# In[10]:


# 1DConv Neural Network

def Conv1D_model():
    
    model = Sequential()
    model.add(Conv1D(64, 
                     #strides=2, 
                     kernel_size=3, activation='relu', 
                     input_shape=(#4,
                                  WINDOWS_SIZE,
                                  1), 
                     name='input_layer'))
    model.add(Conv1D(32, kernel_size=3, activation='relu', name='conv32_layer'))
    model.add(MaxPooling1D(pool_size=1, strides=2, padding='valid', data_format='channels_last'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='tanh')) 
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['MSE'],
    )
    
    return model


# In[ ]:




