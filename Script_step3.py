## Script for generating stats doc with centers of mass and NPY files

## For this version of the script just copy paste all functions into command window##
## Then run this ###


##############
dataset = "TracerX100" 
dataset2 = '/home/chintan/Desktop/TracerX/TracerX100'

##############


import SimpleITK as sitk
import time
from skimage import measure
import numpy as np
from scipy import ndimage
import pandas as pd
import functions
import os


# setup dataframe
columns = ['dataset','patient',  # original
			'pathToData','pathToMask', # '' if files not found
           'stackMin','stackMax',
           'orgSpacX','orgSpacY','orgSpacZ',
           'sizeX','sizeY','sizeZ',
           "voxelCountList", 'chosenMaskIdx', # mask
           'bboxX','bboxY','bboxZ',
           'comX','comY','comZ']

myMainDataFrame = pd.DataFrame(columns=columns)
myMainDataFrame.head()



# 	
# 	
# 	`7MMF'        .g8""8q.     .g8""8q. `7MM"""Mq.
# 	  MM        .dP'    `YM. .dP'    `YM. MM   `MM.
# 	  MM        dM'      `MM dM'      `MM MM   ,M9
# 	  MM        MM        MM MM        MM MMmmdM9
# 	  MM      , MM.      ,MP MM.      ,MP MM
# 	  MM     ,M `Mb.    ,dP' `Mb.    ,dP' MM
# 	.JMMmmmmMMM   `"bmmd"'     `"bmmd"' .JMML.
# 	
# 	

pathToList = dataset + "_links.csv"
dataFrame = pd.DataFrame.from_csv(pathToList, index_col = None)
print "original dataFrame size : " , dataFrame.shape


# just for sanity
dataFrameTemp = dataFrame[ ( dataFrame['pathToData'].isnull() ) | ( dataFrame['pathToMask'].isnull() )  ]
print "after exclusion dataFrame size : " , dataFrameTemp.shape


for i in range (dataFrame.shape[0]):

	pID = str(dataFrame['patient'][i])

	# only process if both data and mask are not null
	if ( pd.isnull( dataFrame['pathToData'][i] )  |  pd.isnull( dataFrame['pathToMask'][i] ) ) == False :

		# time
		start = time.time()

		# load data
		data = sitk.ReadImage( dataFrame['pathToData'][i] )
		# load mask
		mask = sitk.ReadImage( dataFrame['pathToMask'][i] )

		print 'old_size: ', data.GetSize()
		print 'old_spacing', data.GetSpacing()

		# make iso
		dataIso = getIsotrpic(data,sitk.sitkLinear)
		maskIso = getIsotrpic(mask,sitk.sitkNearestNeighbor)

		print 'new_size: ', dataIso.GetSize()
		print 'new_spacing', dataIso.GetSpacing()

		# get disconnected masks
		maskList, maskListVolume = checkNoduleCount ( sitk.GetArrayFromImage(maskIso) )
		print maskListVolume
		# if maskList is not empty
		if maskListVolume != []:
			idx =  np.argmax (maskListVolume)
			print "largest mask: " , idx
			maskToUse = maskList [ idx ]


			# get bbox
			bbox = getBbox(maskToUse)
			print "bbox: " ,bbox

			# centroid
			com = ndimage.measurements.center_of_mass(maskToUse)
			com = getClosestSlice(com)
			print "com: " ,com

			# time
			stop = time.time()

			# save 
			np.save( dataset2 + "_out/data/" + pID  , sitk.GetArrayFromImage(dataIso).astype(np.int16) ) 
			np.save( dataset2 + "_out/mask/" + pID , sitk.GetArrayFromImage(maskIso).astype(np.int8) ) 
			print "time for " , pID , " : " , stop-start
			print " ------------------------------"


			# save into dataFrame
			myMainDataFrame.loc[i] = [dataset,pID,
											dataFrame['pathToData'][i] , dataFrame['pathToMask'][i] ,
			                                sitk.GetArrayFromImage(dataIso).min(),sitk.GetArrayFromImage(dataIso).max(),
			                                dataIso.GetSize()[0],dataIso.GetSize()[1],dataIso.GetSize()[2],
			                                data.GetSpacing()[0],data.GetSpacing()[1],data.GetSpacing()[2],
			                                maskListVolume , idx,
			                                bbox[6],bbox[7],bbox[8],
			                                com[0],com[1],com[2]
			                                ]
		# if it is empty - treat as as nan and even make the pathToMask into nan	                                
		else:
			myMainDataFrame.loc[i] = [dataset,pID,dataFrame['pathToData'][i] , dataFrame['pathToMask'][i] ,'','','','','','','','','','','','','','','','']

	# else one of them is nan
	else:
		myMainDataFrame.loc[i] = [dataset,pID,dataFrame['pathToData'][i] , dataFrame['pathToMask'][i] ,'','','','','','','','','','','','','','','','']

	# after each loop round
	# it will overwrite it everytime..
	myMainDataFrame.to_csv(dataset + '_stats.csv')

# end of loop
