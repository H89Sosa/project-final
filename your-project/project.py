import pandas as pd
import numpy as np
import librosa
import soundfile as sf


import matplotlib.pyplot as plt
import librosa.display
import IPython.display as ipd
import crepe
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

mix, sr = librosa.load('data/aceofspades/mix.mp3')

gtr, sr = librosa.load('data/aceofspades/Guitar.mp3')


test = mix

WINDOWS_SIZE = 5
STRIDE = 1
indexer = np.arange(WINDOWS_SIZE)[None, :] + STRIDE*np.arange(len(test) + STRIDE - WINDOWS_SIZE)[:, None]

y = gtr[:-WINDOWS_SIZE+1]

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D

def create_model():
    
    model = Sequential()
    model.add(Conv1D(64, 
#                    strides=5, 
                     kernel_size=3, activation='relu', 
                     input_shape=(#4,
                                  WINDOWS_SIZE,
                                  ), 
                     name='input_layer'))
    model.add(Conv1D(32, kernel_size=3, activation='relu', name='conv32_layer'))
#     model.add(MaxPooling1D())
#     model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='relu')) 
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    return model

model1D = create_model()

model1D.fit(indexer, gtr)



