
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories

import numpy as np
import pandas as pd
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers
from sklearn.model_selection import train_test_split
import cv2
from random import shuffle
import cnn_sgn
import tensorflowjs as tfjs

IMG_SIZE = 64
LR = 1e-3  #.001 learing rate

nb_classes=29

MODEL_NAME = 'handsign.model'

def one_hot_targets_(labels_dense,nb_classes):
    targets = np.array(labels_dense).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets


train_data = np.load('train_data.npy',encoding="latin1", allow_pickle=True)
# test_data = np.load('test_data.npy',encoding="latin1")

train = train_data[3000:]
test = train_data[:3000]

print('traindatlen:'+str(len(train)))
print('testdatalen:'+str(len(test)))

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]
Y1=one_hot_targets_(Y,nb_classes)
print('val y'+str(Y1))
print('len X:'+str(len(X)))
print('len Y:'+str(len(Y)))
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]
test_y1=one_hot_targets_(test_y,nb_classes)
Y=Y1
print('test_x:'+str(len(test_x)))
print('test_y:'+str(len(test_y)))
print('val y'+str(test_y1))
test_y=test_y1

model =cnn_sgn.create_model()
model.fit(X, Y, batch_size=64, epochs=5, validation_split = 0.1)
tfjs.converters.save_keras_model(model, 'tfjs_files')

model.save(MODEL_NAME)



evaluate_metrics = model.evaluate(test_x, test_y)
print("\nEvaluation Accuracy = ", "{:.2f}%".format(evaluate_metrics[1]*100),
      "\nEvaluation loss = " ,"{:.6f}".format(evaluate_metrics[0]))
