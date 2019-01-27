# Implementation of Behavioral cloning model

import os
import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

samples = [] 

# Load udacity provided training data
with open('/opt/udacity_training_data/data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    next(reader, None) # Skip the first record as it contains the headings
    for line in reader:
        samples.append(line)
        
# TODO:         
# Add any data generated from the simultor here

# Split the data for training and validation
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
#print(len(samples), len(train_samples), len(validation_samples))

import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

#code for generator
def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: 
        shuffle(samples) 
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                    for i in range(0,3): # Using Center, Left and Right camera images                        
                        name = '/opt/udacity_training_data/data/IMG/'+batch_sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) 
                        center_angle = float(batch_sample[3]) # Steering angle
                        images.append(center_image)
                        
                        # Steering angle correction factor for left and right camera.
                        # For left camera image increase the steering angle by 0.2
                        # For left camera image decrease the steering angle by 0.2
                        
                        if(i==0):
                            angles.append(center_angle)
                        elif(i==1):
                            angles.append(center_angle+0.2)
                        elif(i==2):
                            angles.append(center_angle-0.2)
                        
                        # Augment the data by flipping it horiontally
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(center_angle*-1)
                        elif(i==1):
                            angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((center_angle-0.2)*-1)
                        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)             

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

# NVIDIA CNN model 
# (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
model = Sequential()

# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Crop the image to remove irrelevant part of the image
model.add(Cropping2D(cropping=((70,25),(0,0))))           

# Layer 1- Convolution, no of filters- 24, filter size= 5x5, stride= 2x2
model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('relu'))

# Layer 2- Convolution, no of filters - 36, filter size= 5x5, stride= 2x2
model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Layer 3- Convolution, no of filters- 48, filter size= 5x5, stride= 2x2
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Layer 4- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))

# Layer 5- Convolution, no of filters- 64, filter size= 3x3, stride= 1x1
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))

# Flatten image
model.add(Flatten())

# Layer 6 - fully connected layer 1
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Layer 7- fully connected layer 2
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Layer 8- fully connected layer 3
model.add(Dense(10))
model.add(Activation('relu'))

# Layer 9- fully connected layer 4
model.add(Dense(1)) # Output layer 

# the output is the steering angle
# Using "mean squared error" as the  loss function snce this is a regression problem
# Optimier used is 'adam'
model.compile(loss='mse',optimizer='adam')

# fit generator is used here as the number of images are generated by the generator
# no of epochs : 3
model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

# Save the model
model.save('model1.h5')

print('Done! Model Saved!')

# keras method to print the model summary
model.summary()