## This model extracts features and provides predictions for TraceRx dataset using a 3D model


#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from theano import function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from tempfile import TemporaryFile
import pandas as pd
# input image dimensions
img_x, img_y, img_z = 50, 50, 50 # x, y, and z may not necessarily correspond to respective dimensions

ROI = img_x/2
# number of channels
img_channels = 1 # probably don't change this

#number of classes
nb_classes = 2

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

###############
#Define seed points
#SeedsFile = '/home/chintan/Desktop/TracerX/TracerX100/corrupted/Final_TracerX100_stats.csv'
SeedsPts = pd.read_csv(SeedsFile)

path1 = '/home/chintan/Desktop/TracerX/TracerX100_out/data'
path2 = '/home/chintan/Desktop/TracerX/TracerX100_out/NPYCrops'

listing = os.listdir(path2) 
num_samples=size(listing)
print num_samples

#save cropped and reshaped files in a matrix       
imlist = os.listdir(path2)
immatrix = array([np.load(path2+ '/' + im2).flatten()
              for im2 in imlist],'f')
 
img_batch=immatrix.reshape(num_samples,img_channels, img_x,img_y, img_z, 1) #this is the batch of images (num_samples, 1,50,50,50,1)
 
X = img_batch

predDir = '/home/chintan/Desktop/TracerX/TracerX100'
modelFile = (os.path.join(predDir,'120_linear.h5'))  #make sure the model is saved here. 
model = load_model(modelFile)

#predictions using model
pred_results = []
for i in xrange(num_samples):
    predictions = model.predict(X[i])
    prob = softmax(predictions)
    prob_survival = prob[:,:]
    pred_results = np.append(pred_results,[prob_survival])
    print(prob_survival)
    
Y_pred = pred_results.reshape(num_samples,nb_classes)

np.savetxt('TracerX3DPredictions.csv', Y_pred , fmt='%s', delimiter = ',')
   
## Extract features
f1 = K.function([model.layers[0].input, K.learning_phase()], [model.layers[21].output])
f2 = K.function([model.layers[0].input, K.learning_phase()], [model.layers[23].output])

sample_img = X[1,:,:,:].reshape(1,img_x, img_y, img_z, 1)

Features1 = f1([sample_img,0])[0]  # 512-D feature vector
Features2 = f2([sample_img,0])[0]  #216 -D feature vector

#F_Vector = Features.flatten()

num_feat = 512 #256
Feat = np.empty([num_feat,]) #when layer_index =19
for i in xrange(X.shape[0]):
	input_image=X[i,:,:,:].reshape(1,img_x, img_y, img_z,1)
	input_image_aslist= [input_image,0]
	func1out = f1(input_image_aslist)
	features = np.asarray(func1out).flatten()
	Feat = np.concatenate((Feat, features), axis = 0)

Feat = np.asarray(Feat).reshape(X.shape[0]+1,num_feat)
Feat512 = Feat[1:Feat.shape[1],:]


num_feat = 256 #256
Feat = np.empty([num_feat,]) #when layer_index =19
for i in xrange(X.shape[0]):
	input_image=X[i,:,:,:].reshape(1,img_x, img_y, img_z,1)
	input_image_aslist= [input_image,0]
	func1out = f2(input_image_aslist)
	features = np.asarray(func1out).flatten()
	Feat = np.concatenate((Feat, features), axis = 0)

Feat = np.asarray(Feat).reshape(X.shape[0]+1,num_feat)
Feat256 = Feat[1:Feat.shape[1],:]

##Visualize input data
n = X.shape[0]-1
s = n/20

plt.figure(figsize=(20, s))
for i in range(n):
    # display original
    ax = plt.subplot(s, n/s, i + 1)
    plt.imshow(squeeze(X[i])[25,:,:])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
