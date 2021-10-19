# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 00:29:40 2021

@author: Areeba
"""

from mtcnn.mtcnn import MTCNN
import os
import cv2
import numpy as np
from keras.models import load_model
from numpy import savez_compressed


#using this to extract single face
def extract_face(filename, required_size = (160,160)):
    im = cv2.imread(filename) #read image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB ) 
    detector = MTCNN() 
    results = detector.detect_faces(im)
    x1, y1, width, height = results[0]['box'] 
    x2 = abs(x1) + width
    y2 = abs(y1) + height
    #extract face
    face_array = im[y1:y2, x1:x2] #croping
    face_array = cv2.resize(face_array, required_size) #resizing to required size so as to sned it into facenet model
    return face_array

### load faces from the directory
def load_faces(directory): #this will take the sub dir as input and list the names of the files in it and then add the path to dir to get the full path of image i order to read 
    faces = list()
    #go through each file 
    for filename in os.listdir(directory):
        path = directory + filename
        face = extract_face(path)
        faces.append(face)
    return faces

#load data that contains one subdirectory for each class that contains images
def load_dataset(directory):
    X, y = list(), list()
    for subdir in os.listdir(directory):
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not os.path.isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for i in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return np.array(X), np.array(y)

#load data
tr_path = './Dataset/'
trainx, trainy = load_dataset(tr_path)
print('Dataset Loaded")

#loda the pretrained facenet model
facenet = load_model('facenet_keras.h5')

def get_embeddings(pixels, model):
    pixels = pixels.astype('float32')
    #standarzie the face piels
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std
    #single sample create
    samples = np.expand_dims(pixels, axis = 0)
    embd = model.predict(samples)
    return embd[0]

#create embeddings of dataset    
new_trainx = list()
for facepixels in trainx:
    embeddings = get_embeddings(facepixels, facenet)
    new_trainx.append(embeddings)
new_trainx = np.array(new_trainx)    


print(new_trainx.shape)

#save the arrays in compressed numpy array format
savez_compressed('5embeddings.npz', new_trainx, trainy)
