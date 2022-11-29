from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import scipy
import pickle

trainingImagePath = '/home/jay-bhagiya/Codes/mini-project/final-codes/face_images/training'

train_datagen = ImageDataGenerator(
				shear_range=0.1,
				zoom_range=0.1,
				horizontal_flip=True)

test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory(
	trainingImagePath,
	target_size=(64, 64),
	batch_size=32,
	class_mode='categorical')

test_set = test_datagen.flow_from_directory(
	trainingImagePath,
	target_size=(64, 64),
	batch_size=32,
	class_mode='categorical')

train_classes = training_set.class_indices

ResultMap = {}
for faceValue, faceName in zip(train_classes.values(), train_classes.keys()):
	ResultMap[faceValue]=faceName

# Saving the face map for future reference 
with open("ResultMap.pkl", 'wb') as fileWriteStream:
	pickle.dump(ResultMap, fileWriteStream)

print("Mapping of Face and its ID", ResultMap)

OutputNeurons=len(ResultMap)
print("\nThe number of output neurons: ", OutputNeurons)

# Initializing the Convolutional Neural Network
classifier = Sequential()
 
# STEP--1 Convolution
# Adding the first layer of CNN
# we are using the format (64,64,3) because we are using TensorFlow backend
# It means 3 matrix of size (64X64) pixels representing Red, Green and Blue components of pixels

classifier.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), input_shape=(64,64,3), activation='relu'))
 
# STEP--2 MAX Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))
 
# ############## ADDITIONAL LAYER of CONVOLUTION for better accuracy #################
classifier.add(Convolution2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
 
classifier.add(MaxPool2D(pool_size=(2,2)))
 
# STEP--3 FLattening
classifier.add(Flatten())
 
# STEP--4 Fully Connected Neural Network
classifier.add(Dense(64, activation='relu'))
 
classifier.add(Dense(OutputNeurons, activation='softmax'))
 
# Compiling the CNN
#classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=["accuracy"])

###########################################################

TRAIN_STEPS_PER_EPOCH = np.ceil((244*0.8/32)-1)
# to ensure that there are enough images for training bahch
VAL_STEPS_PER_EPOCH = np.ceil((244*0.2/32)-1)
 
###########################################################
import time
# Measuring the time taken by the model to train
StartTime=time.time()
 
# Starting the model training
classifier.fit(
			training_set,
			steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
			epochs=10,
			validation_data=test_set,
			validation_steps=VAL_STEPS_PER_EPOCH
		)

classifier.save('model.h5')

EndTime=time.time()
print("###### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ######')