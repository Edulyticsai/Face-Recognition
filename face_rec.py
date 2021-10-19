# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 01:59:11 2021
FACE-RECOGNITION
@author: Areeba
"""

# In[1]:


#Importing nececessary packages

import warnings
warnings.filterwarnings('ignore')
import imutils 
import argparse
from imutils.video import VideoStream
import numpy as np 
import time
import cv2
from mtcnn.mtcnn import MTCNN
from retinaface import RetinaFace
from sklearn.preprocessing import LabelEncoder, Normalizer
from keras.models import load_model
import math
import pandas

# In[2]:


#Load the encoding 
embd_data = np.load('5embeddings.npz',allow_pickle=True)
trainx_embd, trainy = embd_data['arr_0'], embd_data['arr_1']
le = LabelEncoder()

facenet = load_model('facenet_keras.h5', compile =  False)
print('Facenet loaded')
norm = Normalizer('l2')


# In[3]:

def distance(embeddings1, embeddings2, distance_metric = 1):
    if distance_metric == 0:
        'taking euclidean distance :'
        dist = np.linalg.norm(embeddings1 - embeddings2)
    elif distance_metric == 1:
        'taking cosine distance :'
        dot = np.sum(np.multiply(embeddings1, embeddings2))
        denominator = np.multiply(np.linalg.norm(embeddings1), np.linalg.norm(embeddings2) )
        similarity = dot/denominator
        dist = np.arccos(similarity) / math.pi 
    return dist


# In[4]:
     
video_capture = cv2.VideoCapture(0)
frame_width = int(video_capture.get(3))
frame_height = int(video_capture.get(4))
size = (frame_width, frame_height)
out = cv2.VideoWriter('outputfile.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
frame_num=0
while True:
    ret, frame = video_capture.read()
    frame_num += 1
    print(frame_num)
    try:
        # frame = cv2.flip(frame, 1 ) 
        detection = RetinaFace.detect_faces(frame)
        for face in detection.keys():
            x1 = detection[face]['facial_area'][0]
            y1 = detection[face]['facial_area'][1]
            x2 = detection[face]['facial_area'][2]
            y2 = detection[face]['facial_area'][3]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
            faces = frame[y1:y2, x1:x2]  
            faces = cv2.resize(faces, (160, 160))
            face_pixels = faces.astype('float32')
            mean, std = face_pixels.mean(), face_pixels.std()
            face_pixels = (face_pixels - mean)/std
            face_pixels = np.expand_dims(face_pixels, axis = 0)
            embd = facenet.predict(face_pixels)
            min_dist = 100 
            for i in range(trainx_embd.shape[0]):
                actual_name = trainy[i]
                dist = distance(trainx_embd[i].reshape(-1,1) , embd.reshape(-1,1), 1 )
                if dist < min_dist:
                    min_dist = dist
                    identity = actual_name 
            if min_dist < 0.29:
                cv2.putText(frame, "Face : " + identity, (x1, y1 - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
                cv2.putText(frame, "Dist : " + str(min_dist), (x1, y1 - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Unknown', (x1, y1 - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            # ims= cv2.resize(frame,(960,540))    
            # cv2.imshow('face_rec', ims)
        out.write(frame)
    except Exception as error:
      print('Face Not detected. Error {}'.format(error))
    
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
video_capture.release()
result.release()
cv2.destroyAllWindows()
print("The video was successfully saved")
