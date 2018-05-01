# run this file second. Can also just do in command window

import os
import pandas as pd

def tracerX100(dirName,_dir):



	pathToData = ''
	pathToMask = ''
	# assuming it exists
	dataFlag = True
	maskFlag = True



	# D:\LUNGS\lung2_in\2097778\ (this only happens in lung1, but will keep it here)
	# check for multiple timepoints indicated by the choice of __dir
	temp = dirName + _dir + "/"
	# if none, break it
	if os.listdir(  temp ) == []:
		dataFlag = False
		maskFlag = False
		# break it
	# if one , take it
	elif len ( os.listdir( temp ) ) == 1:
		__dir = os.listdir( temp )[0]
	# if more than one, take most recent (but not underscore)
	else:
		sortedList = sorted( os.listdir( temp ) )
		# if underscore take second to last
		if sortedList[-1] == '_':
			__dir = sortedList[-2]
		else:
			__dir = sortedList[-1]


	# if they havent been turned into false from above
	if dataFlag and maskFlag:
		# "D:\LUNGS\lung1_in\62999\62999_20100426_\
		for ___dir in os.listdir( dirName + _dir + "/" + __dir + "/" ):

			#
			# DATA
			#
			if "Reconstructions" in ___dir:
				# D:\LUNGS\lung2_in\2097778\2097778_20060410\2097778_20060410_Reconstructions
				temp = dirName + _dir + "/" + __dir + "/" + ___dir
				# no data file
				if os.listdir(  temp ) == []:
					dataFlag = False
				# if one and ends with nrrd
				elif len ( os.listdir(  temp ) ) == 1 and  [ file for file in os.listdir( temp ) ][0].endswith(".nrrd") :
					pathToData = temp + "/" + [ file for file in os.listdir( temp ) ][0]
				# if one and not nrrd
				elif len ( os.listdir(  temp ) ) == 1 and not [ file for file in os.listdir( temp ) ][0].endswith(".nrrd") :
					dataFlag = False
					print "one and not nrrd"
				# if more than one - if more than one, one has to be nrrd
				else:
					for dataFile in os.listdir( temp ):
						if dataFile.endswith("nrrd"):
							pathToData = temp + "/" + dataFile  

			#
			# MASK
			#
			if "Segmentations" in ___dir:
				# if file exists
				# lung2_in\0012356\0012356_20051201\0012356_20051201_Segmentations_old
				temp = dirName + _dir + "/" + __dir + "/" + ___dir
				# if empty
				if os.listdir( temp ) == []:
					maskFlag = False
				# take first file if not empty
				else:
					pathToMask = temp + "/" + os.listdir(  temp )[0]
					



	return pathToData , pathToMask
	
	
	
dataset = "TracerX100" 
dirName = "/home/chintan/Desktop/TracerX/TracerX100/Curated/"  
#############
counter = 0
columns = ['dataset','patient',  # original
			'pathToData','pathToMask']

myMainDataFrame = pd.DataFrame(columns=columns)
myMainDataFrame.head()


for _dir in os.listdir( dirName ): 
	# get patient id
	pID =  _dir
	#########################################################
	pathToData , pathToMask = tracerX100(dirName,_dir) 
	#########################################################
	# save into dataFrame
	myMainDataFrame.loc[counter] = [dataset,pID,
									pathToData,pathToMask]
	counter +=1
	# it will overwrite it everytime..
	myMainDataFrame.to_csv(dataset + '_links.csv')
