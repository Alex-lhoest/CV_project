from keras.models import load_model
import numpy as np
from PIL import Image
import keras

from unet import mean_iou

import sys

dependencies = {
    'mean_iou': mean_iou
}
# Load the models
model = load_model('../models/300epochs/unet_weights.34-0.01.h5', custom_objects=dependencies)

# Load the image
image_inp = Image.open('../eval/images/input/elps_eye_205.png')
# image_inp = Image.open('../NoEllipses/noelps_eye2015-01-16_02-46-45-005.png')

# Scale the input between [0 1] and change the arrays (1, 512, 512, 3)
image_inp = np.array(image_inp)/255.
inp = np.expand_dims(image_inp, axis=0)

# Predict the segmanted mask given the model
y_pred = model.predict(inp)

# Reshape the output (240, 320)
y_pred = np.reshape(y_pred, (240, 320))


# Set all pixels under a threshold at 0 and the other at 1
# This threshold is computed manually 
# y_pred[y_pred < 0.1] = 0
# y_pred[y_pred >= 0.1] = 1

# Add the segmantation to the input image
# Red channel is used for the roads and the Green one for the buildings
image_inp[..., 0] += y_pred

image_inp[image_inp > 1] = 1

# Transform the array in a image representation
image_out = keras.preprocessing.image.array_to_img(image_inp)
image_out.show()

# Save the image
# image_out.save("nice.png", format='png')