# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:58:28 2019

@author: IlknurKaraduman
"""
#routers.py dosyasında with np.load(path, allow_pickle=True) as f: olarak değişiklik yaptım
#ve ayrıca keras işlemlerinin başlarına tensorflow. ekledim çünkü  hata çıktı. sürüm: tf 2.0.0
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

from tensorflow.keras.utils import plot_model

from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing import sequence

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, SimpleRNN

max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)


print('Loading data...')
(input_train, y_train), (input_test, y_test) = reuters.load_data(
        num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)


from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(46, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop', 
              loss='sparse_categorical_crossentropy', 
              metrics=['acc'])

history = model.fit(input_train, y_train,
                    epochs=5,
                    batch_size=128,
                    validation_split=0.2)