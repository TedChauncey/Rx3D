#Crop large npys around center of mass
#Taf Chaunzwa 4/30/18

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import theano
from PIL import Image
from numpy import *
import pandas as pd

def centerAndNormalize(arr):
    out = arr
    oldMin = -1024.
    oldRange = 3071.+1024.
    newRange = 1.
    newMin = 0.
    sikoAll = ((( out  - oldMin) * newRange) / oldRange) + newMin
    return sikoAll
    
# input image dimensions
img_x, img_y, img_z = 50, 50, 50 # x, y, and z may not necessarily correspond to respective dimensions

ROI = img_x/2
# number of channels
img_channels = 1 # probably don't change this    
    
#Define seed points
#SeedsFile = '/home/chintan/Desktop/TracerX/TracerX100/corrupted/First25.csv'
SeedsFile = '/home/chintan/Desktop/TracerX/TracerX100/TracerX100_2ndbatch_stats.csv'
SeedPts = pd.read_csv(SeedsFile)
comX = pd.Series.as_matrix(SeedPts.loc[:,'comX'])   
comY = pd.Series.as_matrix(SeedPts.loc[:,'comY'])
comZ = pd.Series.as_matrix(SeedPts.loc[:,'comZ'])

path1 = '/home/chintan/Desktop/TracerX/TracerX100_out/data'
path2 = '/home/chintan/Desktop/TracerX/TracerX100_out/NPYCrops'

listing = os.listdir(path1) 
num_samples=size(listing)
print ('number of samples:', num_samples)
listing =sort(listing).tolist()

for file in listing:
    im = np.load(path1 + '/' + file)
    i = listing.index(file)
    img = centerAndNormalize(im)
    img_crop = img[comX[i]-ROI:comX[i]+ROI,comY[i]-ROI:comY[i]+ROI,comZ[i]-ROI:comZ[i]+ROI]
    print (i)
    print (file) 
    img_input = img_crop.reshape(img_channels,img_x,img_y,img_z,1)
    np.save(path2+'/'+file, img_input)       
