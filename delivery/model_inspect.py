import csv
import cv2
import numpy as np
import sklearn
import keras.models
from keras.models import Model
from keras import backend as K
from PIL import Image


def _8bit_image_2(x):
    mmin, mmax = [ np.amin(x), np.amax(x) ]
    range = mmax - mmin

    return (((x+mmin) / range)*255.0).astype('uint8')

model = keras.models.load_model("working_model.h5")

im = cv2.imread('examples/center_driving.jpg')
im2 = cv2.imread('examples/after_bridge_driving.jpg')
im3 = cv2.imread('examples/veer_right.jpg')
data = np.array([im3])

layer_id = 2
get_layer_output = K.function([model.layers[0].input], [model.layers[layer_id].output])


output = get_layer_output( [data] )[0]
print ("layer {0} with {1} outputs of shape: {2}".format(layer_id, output.shape[3], output[0,:,:,0].shape))
for i in range(output.shape[3]):
    imga = output[0,:,:,i]
    img = Image.fromarray( _8bit_image_2(imga) )
    img.save("vis/layer{0}_{1}.png".format(layer_id, i))

