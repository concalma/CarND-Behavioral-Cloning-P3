import csv
import cv2
import numpy as np



lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

def current_path(csvpath):
	filename = csvpath.split('/')[-1]
	return 'data/IMG/' + filename


use_side_cameras = True
images = []
measurements = []
for line in lines:
	correction = 0.1
	steering_center = float(line[3])
	images.append( cv2.imread(current_path(line[0])) )
	measurements.append(steering_center)

	if use_side_cameras:
		images.append( cv2.imread(current_path(line[1])) )
		images.append( cv2.imread(current_path(line[2])) )

		measurements.append(steering_center + correction )
		measurements.append(steering_center - correction )
	

	


#adding flipped images
aug_images, aug_measurements = [] ,[]
for image, measurement in zip(images, measurements):
	aug_images.append( image )
	aug_measurements.append( measurement) 
	aug_images.append( cv2.flip(image, 1) ) 
	aug_measurements.append( -1.0 * measurement )

X_train = np.array(aug_images)
y_train = np.array(aug_measurements)

print(X_train.shape)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Lambda,  Dense, Dropout, Activation
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Cropping2D
from keras.callbacks import TensorBoard


tensorflowcb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
model = Sequential()

#normalization
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add( Cropping2D(cropping=((70,25),(0,0))))

def lenet_model(model):
	model.add( Convolution2D( 6, 5, 5, border_mode='valid', activation='relu'  ))
	model.add( MaxPooling2D() )
	model.add( Convolution2D( 6, 5, 5, border_mode='valid', activation='relu'  ))
	model.add( MaxPooling2D() )
	model.add( Flatten())
	model.add( Dense(120) )
	model.add( Dense(84) )
	model.add( Dense(1) )
	return model

def nvidia_model(model):
	model.add( Convolution2D( 24, 5, 5, subsample=(2,2), border_mode = 'valid', activation='relu' ))
	model.add( Convolution2D( 36, 5, 5, subsample=(2,2), border_mode = 'valid', activation = 'relu' ))
	model.add( Convolution2D( 48, 5, 5, subsample=(2,2), border_mode = 'valid', activation = 'relu' ))
	model.add( Convolution2D( 64, 3, 3, border_mode = 'valid', activation = 'relu' ))
	model.add( Convolution2D( 64, 3, 3, border_mode = 'valid', activation = 'relu' ))

	model.add( Flatten() )

	model.add( Dropout(0.5) )
	model.add( Dense(1164 ))
	model.add( Dropout(0.5) )
	model.add( Dense(100 ))
	model.add( Dropout(0.5) )
	model.add( Dense(50))
	model.add( Dropout(0.5) )
	model.add( Dense(10))
	model.add( Dropout(0.5) )
	model.add( Dense(1))
	return model


model = nvidia_model(model)






model.compile(loss='mse', optimizer='adam' )
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, callbacks=[tensorflowcb] ) 


model.save('model.h5')
