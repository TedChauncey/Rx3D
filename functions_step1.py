## run this script before starting process. 
## TC 4.30.18

import SimpleITK as sitk
from skimage import measure
import numpy as np
from scipy import ndimage
import os

# data to make isotropic
# sitk.sitkLinear OR sitk.sitkNearestNeighbor 

def getIsotrpic(data,interpMethod):
    original_spacing = data.GetSpacing()
    original_size = data.GetSize()
    new_spacing = [1, 1, 1]
    new_size = [int(round(original_size[0]*(original_spacing[0]))), 
                int(round(original_size[1]*(original_spacing[1]))),
                int(round(original_size[2]*(original_spacing[2])))]

    resampleImageFilter = sitk.ResampleImageFilter()
    data_iso = resampleImageFilter.Execute(data, 
                                            new_size, 
                                            sitk.Transform(), 
                                            interpMethod, 
                                            data.GetOrigin(),
                                            new_spacing, 
                                            data.GetDirection(),
                                            0, 
                                            data.GetPixelIDValue())


    return data_iso

def checkNoduleCount (maskData):
    # copy for safety
    copyMaskData = maskData
    # get blobs
    all_labels = measure.label(copyMaskData)
    # get number of blobs ( inlcuding background as zero
    noduleCount = all_labels.max()
    #
    # list to populate
    maskDataList = []
    maskDataListVolume = []
    # supress one mask and leave the others
    # 0 is the background - so dont deal with it
    for label in range ( 1, noduleCount + 1):
        
        # make an array of zeros
        tempMaskData = np.zeros((copyMaskData.shape[0],copyMaskData.shape[1],copyMaskData.shape[2]) , dtype=np.int64 )
        #
        arrays = np.where(all_labels == label)
        # just loop through lenght of one of the 3 arrays - they are all the same
        for k in range ( len(arrays[0]) ):
            tempMaskData[ arrays[0][k] ][ arrays[1][k] ][ arrays[2][k] ] = 1.0

        # when array done
        maskDataList.append(tempMaskData)
        maskDataListVolume.append( tempMaskData.sum() )
        
    return maskDataList,maskDataListVolume


def getClosestSlice (centroidSciPy):
    return int( centroidSciPy[0] ),int( centroidSciPy[1] ),int( centroidSciPy[2] )


def getBbox(maskData):
    # crop maskData to only the 1's  
    # http://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    # maskData order is z,y,x because we already rolled it
    
    Z = np.any(maskData, axis=(1, 2))
    Y = np.any(maskData, axis=(0, 2))
    X = np.any(maskData, axis=(0, 1))

    Xmin, Xmax = np.where(X)[0][[0, -1]]
    Ymin, Ymax = np.where(Y)[0][[0, -1]]
    Zmin, Zmax = np.where(Z)[0][[0, -1]]
    

    return Xmin, Xmax, Ymin, Ymax, Zmin, Zmax, Xmax-Xmin,   Ymax-Ymin,   Zmax-Zmin 
