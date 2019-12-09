from keras.models import load_model
import numpy as np
from PIL import Image
import keras

# Load the models
model_r = load_model('road_final_model_512.h5')
model_b = load_model('building_final_model_512.h5')

# Load the image
image_inp = Image.open('../../data/roads/resized_image/Test/Image/Test_input/image_8.tiff')

# Scale the input between [0 1] and change the arrays (1, 512, 512, 3)
image_inp = np.array(image_inp)/255.
inp = np.expand_dims(image_inp, axis=0)

# Predict the segmanted mask given the model
y_pred_r = model_r.predict(inp)
y_pred_b = model_b.predict(inp)

# Reshape the output (512, 512)
y_pred_r = np.reshape(y_pred_r, (512, 512))
y_pred_b = np.reshape(y_pred_b, (512, 512))

# Set all pixels under a threshold at 0 and the other at 1
# This threshold is computed manually 
y_pred_r[y_pred_r < 0.1] = 0
y_pred_r[y_pred_r >= 0.1] = 1

y_pred_b[y_pred_b < 0.5] = 0
y_pred_b[y_pred_b >= 0.5] = 1

# Add the segmantation to the input image
# Red channel is used for the roads and the Green one for the buildings
image_inp[..., 0] += y_pred_r
image_inp[..., 1] += y_pred_b


image_inp[image_inp > 1] = 1

# Transform the array in a image representation
image_out = keras.preprocessing.image.array_to_img(image_inp)
image_out.show()

# Save the image
# image_out.save("nice.png", format='png')