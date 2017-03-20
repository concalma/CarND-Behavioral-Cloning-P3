import csv
import cv2
import numpy as np
import sklearn


################################################################################
#           original training CSV data load                 
################################################################################
lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

################################################################################
#           python generator logic to load images on the fly + augmentation logic
################################################################################
from sklearn.model_selection import train_test_split
from random import shuffle
train_samples, validation_samples = train_test_split( lines, test_size=0.2 )

print("train_count: ", len(train_samples), "valid_count", len(validation_samples))
def current_path(csvpath):
	filename = csvpath.split('/')[-1]
	return 'data/IMG/' + filename

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[ offset: offset+batch_size ]

            images = []
            steer = []

            for batch_sample in batch_samples:
                center_image = cv2.imread( current_path( batch_sample[0] ))
                left_image =   cv2.imread( current_path( batch_sample[1] ))
                right_image =  cv2.imread( current_path( batch_sample[2] ))

                correction = 0.1
                steering_center = float( batch_sample[3] )
                steering_leftcam = steering_center + correction
                steering_rightcam = steering_center - correction

                images += [center_image, left_image, right_image]
                steer += [steering_center, steering_leftcam, steering_rightcam]

            aug_images, aug_measurements = [] ,[]
            for image, measurement in zip(images, steer):
                aug_images.append( image )
                aug_measurements.append( measurement ) 
                aug_images.append( cv2.flip(image, 1) ) 
                aug_measurements.append( -1.0 * measurement )

            X_train = np.array(aug_images)
            y_train = np.array(aug_measurements)

            # print(X_train.shape, y_train_shape)
            yield sklearn.utils.shuffle( X_train, y_train )

augmentation_factor = 6 
samples_per_epoch = len(train_samples)   * augmentation_factor  # 3x * 2
nb_val_samples = len(validation_samples) * augmentation_factor  # 3x * 2

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

from keras.models import Sequential
from keras.layers import Flatten, Lambda,  Dense, Dropout, Activation
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Cropping2D
from keras.callbacks import TensorBoard


tensorflowcb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
model = Sequential()

################################################################################
#       normalization
################################################################################
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add( Cropping2D(cropping=((70,25),(0,0))))

keep_prob = 0.5

################################################################################
#       models definition
################################################################################
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

def nvidialite_model(model):
	model.add( Convolution2D( 48, 5, 5, subsample=(2,2), border_mode = 'valid', activation='relu' ))
	model.add( Convolution2D( 64, 5, 5, subsample=(2,2), border_mode = 'valid', activation = 'relu' ))

	model.add( Flatten() )

	model.add( Dense(100, activation='relu' ))
	model.add( Dropout(keep_prob) )
	model.add( Dense(50, activation='relu' ))
	model.add( Dropout(keep_prob) )
	model.add( Dense(10, activation='relu' ))
	model.add( Dense(1, activation='tanh' ))
	return model
        
def nvidia_model(model):
	model.add( Convolution2D( 24, 5, 5, subsample=(2,2), border_mode = 'valid', activation='relu' ))
	model.add( Convolution2D( 36, 5, 5, subsample=(2,2), border_mode = 'valid', activation = 'relu' ))
	model.add( Convolution2D( 48, 5, 5, subsample=(2,2), border_mode = 'valid', activation = 'relu' ))
	model.add( Convolution2D( 64, 3, 3, border_mode = 'valid', activation = 'relu' ))
	model.add( Convolution2D( 64, 3, 3, border_mode = 'valid', activation = 'relu' ))

	model.add( Flatten() )

	model.add( Dropout(keep_prob) ) 
	model.add( Dense(1164, activation='relu' ))
	model.add( Dropout(keep_prob) )
	model.add( Dense(100, activation='relu' ))
	model.add( Dropout(keep_prob) )
	model.add( Dense(50, activation='relu' ))
	model.add( Dropout(keep_prob) )
	model.add( Dense(10, activation='relu' ))
	model.add( Dense(1, activation='tanh' ))
	return model


################################################################################
#       keras run         
################################################################################
model = nvidia_model(model)
model.compile(loss='mse', optimizer='adam' )

model.fit_generator( train_generator, samples_per_epoch=samples_per_epoch, 
                    validation_data=validation_generator, nb_val_samples=nb_val_samples, 
                    nb_epoch=3)

model.save('model.h5')
