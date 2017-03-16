import csv
import cv2
import numpy as np



lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = 'data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	m = float(line[3])
	measurements.append(m)


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
	model.add( Dense(100) )
	model.add( Dense(50) )
	model.add( Dense(10) )
	model.add( Dense(1) )
	return model


model = nvidia_model(model)






model.compile(loss='mse', optimizer='adam', lr=0.001)
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5) 


model.save('model.h5')
