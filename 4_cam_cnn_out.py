import numpy as np
import pandas as pd
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras import regularizers
from sklearn.model_selection import train_test_split
import cv2
from random import shuffle
import imutils

from collections import Counter


import time

import cnn_sgn

IMG_SIZE = 64
LR = 1e-3

nb_classes=28

MODEL_NAME = 'handsign.model'



model=cnn_sgn.create_model()

model = load_model(MODEL_NAME)
print('model loaded!')




# organize imports
import cv2
import imutils
import numpy as np

from collections import Counter

import time
import os
from twilio.rest import Client


# Your Account Sid and Auth Token from twilio.com/console
# and set the environment variables. See http://twil.io/secure

client = Client(account_sid, auth_token)

last_letter = ""

print("Enter your name: ")

name = input()

def send_message(message):
    message = client.chat.services('IS470d35d9675049c5b9d9e78b7dc778ea') \
                 .channels('CH8f8cab0dc3394cd58b952d604b6537fb') \
                 .messages \
                 .create(from_=name, body=message)



           #0    1    2    3        4        5    6    7    8    9    10   11   12   13   14     15       16   17   18   19   20        21        22   23   24   25   26   27  28
out_label=['A', 'B', 'C', 'D', 'backspace', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'blank space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ]


pre=[]
s=''
camera = cv2.VideoCapture(0)
top, right, bottom, left = 210, 0, 460, 250
num_frames = 0
last_added=''
prev_label=''
count=0
text=[]
last_word = ""

while(True):
    (grabbed, frame) = camera.read()

    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    

    (height, width) = frame.shape[:2]
    roi = frame[top:bottom, right:left]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Current Roi', gray)
    img = cv2.resize(gray, (IMG_SIZE,IMG_SIZE))
    data = img.reshape(1,IMG_SIZE,IMG_SIZE,1)
    
    model_out = model.predict([data])[0]
    if max(model_out)*100 > 80:
        pnb=np.argmax(model_out)
        print(str(np.argmax(model_out))+" "+str(out_label[pnb]))
        cv2.putText(clone,
           '%s ' % (str(out_label[pnb])),
           (0, 200), cv2.FONT_HERSHEY_PLAIN,3,(255, 255, 255))
        cur_label=str(out_label[pnb])
        if(cur_label==prev_label):
            count += 1
        else:
            count=0
        prev_label=str(out_label[pnb])
        if(count>=30):
            if (cur_label == 'nothing' and last_added == 'nothing'):
                word1 = ''.join(map(str, text))
                print(word1)
                text = []
                text1 = []
                count = 0
                word1.replace("_", " ")
                if word1:
                    send_message(word1)
            else:
                if(cur_label=='nothing'):
                    text=text
                elif(cur_label=='backspace'):
                    text=text[:-1]
                elif(cur_label=='blank space'):
                    text.append('_')
                else:
                    text.append(cur_label)
                last_added=cur_label
                count=0
    cv2.rectangle(clone, (right, top), (left, bottom), (255,255,255), 2)
    listToStr = ''.join(map(str, text)) 
    cv2.putText(clone,'%s ' % (listToStr),(0, 150), cv2.FONT_HERSHEY_PLAIN,5,(255, 255, 255))

    cv2.putText(clone, '%s ' % (str(s)),(10, 60), cv2.FONT_HERSHEY_PLAIN,3,(0, 0, 0))

    num_frames += 1
    cv2.imshow("Video Feed", clone)
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
        break

    elif keypress == 27:
        break

camera.release()
cv2.destroyAllWindows()
