# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:08:51 2022

@author: nikil
"""

import os
import argparse
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm 
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image 
import imutils
import math
from keras.utils.np_utils import to_categorical
import h5py
import argparse
#from keras.models import load_model
from keras.utils import plot_model
from keras.models import Model,load_model,Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Conv2D, MaxPool2D,  Dropout,Flatten, Input, concatenate, Dense
from keras.datasets import mnist,cifar100
from keras.preprocessing.image import Iterator
import keras.backend as K


IMG_SIZE = 96

def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)
    #return abs(x - y)


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))

def prediction(x,y,model):
    
    img = cv2.imread(args.img_path,cv2.IMREAD_GRAYSCALE)
    rot = cv2.imread(args.rot_img_path,cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
    rot = cv2.resize(rot, (IMG_SIZE,IMG_SIZE))
    print('display original image')
    plt.imshow(img)
    plt.show()
    print('display rotated image')
    plt.imshow(rot)
    plt.show()

    tempp=(np.array(img))/255
    tempp2=(np.array(rot))/255
    tempp = tempp.reshape(-1,IMG_SIZE,IMG_SIZE,1)
    tempp2 = tempp2.reshape(-1,IMG_SIZE,IMG_SIZE,1)
    predictions = model.predict_on_batch([tempp,tempp2])
    predicted_angles = np.argmax(predictions, axis=1)
    idx = (-predictions).argsort()[0][:]
    sum = predictions[0][idx[0]]+predictions[0][idx[1]]
    xx=(((idx[0]*10)*(predictions[0][idx[0]]/sum))+((idx[1]*10)*(predictions[0][idx[1]]/sum)))
    print('the raw angle between the two images is:',predicted_angles*10)
    print('The averaged out error between the images is ', xx)





model = load_model('models/rotnet_11.hdf5', custom_objects={'angle_error': angle_error})

parser = argparse.ArgumentParser(description='Inference program for deetcting ablgke of rotation between 2 images')


parser.add_argument('--img_path', type=str, help='This is the original image')
parser.add_argument('--rot_img_path', type=str, help='this is the rortated image ')

args = parser.parse_args()
print("img",args.img_path)  
print("rot",args.rot_img_path)

prediction(args.img_path,args.rot_img_path,model)
