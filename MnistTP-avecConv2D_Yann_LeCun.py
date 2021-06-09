# -*- coding: utf-8 -*-
"""
Created on Wed Feb 03 19:31:24 2021

@author: Romain Marie
"""

from keras.datasets import mnist
from keras.utils import np_utils,plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import matplotlib

# Pour utiliser tensorflow
from keras import backend as K
K.set_image_data_format('channels_last')

# On charge MNIST
(X_train, Y_train), (X_test,Y_test) = mnist.load_data()

# On prépare les données pour qu'elles correspondent à ce qui est attendu
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) # NHWC
X_train = X_train.astype('float32')/255 # [0,256[ -> [0f,1f[

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test = X_test.astype('float32')/255

Y_train = np_utils.to_categorical(Y_train,10) # 4 -> [0 0 0 0 1 0 0 0 0 0]
Y_test = np_utils.to_categorical(Y_test,10)
 
     
# Modèle 1 : celui du TP 2
# 1 couche cachée de 32 neurones (fully connected)
model2 = Sequential()
# 1 couche cachée de 6 neurones (fully connected)
model2.add(Convolution2D(6,(5,5),padding='same',  activation="relu",input_shape=(28,28,1)))
model2.add(MaxPooling2D(pool_size=(2,2)))
# 1 couche cachée de 16 neurones (fully connected)
model2.add(Convolution2D(16,(5,5), activation="relu", input_shape=(14,14,6) ))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Flatten(input_shape=(5,5,16)))
model2.add(Dense(120,activation='sigmoid'))
model2.add(Dense(84,activation='sigmoid'))
model2.add(Dense(10,activation='softmax'))
model2.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
model2.summary()
 
h1 = model2.fit(X_train, Y_train, 
          batch_size=32,epochs=30, verbose=1,validation_data=(X_test,Y_test))
model2.save('modele_TP2_avecConv2D_Yann_LeCun.h5');
matplotlib.pyplot.plot(h1.history['val_accuracy'])
matplotlib.pyplot.plot(h1.history['accuracy'])