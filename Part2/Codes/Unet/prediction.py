from keras.models import load_model
import numpy as np
from PIL import Image
import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from unet import mean_iou

import sys

dependencies = {
    'mean_iou': mean_iou
}
# # Load the models
# model = load_model('unet_weights.39-0.01.h5')

# # Load the image
# image_inp = Image.open('../../test/images/input/elps_eye_4.png')
# mask_inp = Image.open('../../test/masks/output/mask_eye_4.png').convert("L")
# # image_inp = Image.open('../../NoEllipses/noelps_eye2015-01-16_02-46-45-005.png')

# mask_inp = np.array(mask_inp)/255.

# # Scale the input between [0 1] and change the arrays (1, 240, 320, 3)
# image_inp = np.array(image_inp)/255.
# inp = np.expand_dims(image_inp, axis=0)

# # Predict the segmanted mask given the model
# y_pred = model.predict(inp, verbose=0)

# # Reshape the output (240, 320)
# y_pred = np.reshape(y_pred, (240, 320))

# # Set all pixels under a threshold at 0 and the other at 1
# # This threshold is computed manually 

# thresh = np.arange(0.0, 1.0, 0.1)

# IoU = []

# for threshold in thresh:

#     y = y_pred.copy()
#     y[y < threshold] = 0
#     y[y >= threshold] = 1


#     union = []
#     inter = []

#     for i in range(240):
#         for j in range(320):
#             if mask_inp[i][j] == 1 or y[i][j] == 1:
#                 union.append(1)
#             if mask_inp[i][j] == 1 and y[i][j] == 1:
#                 inter.append(1)

#     IoU.append(len(inter)/(len(union)))
#     print("IoU = ", IoU[-1])



# image_inp[..., 0] += y_pred

# image_inp[image_inp > 1] = 1

# # Transform the array in a image representation
# image_out = keras.preprocessing.image.array_to_img(image_inp)
# image_out.show()

# # Save the image
# # image_out.save("nice.png", format='png')

def make_prediction():

    model = load_model('unet_weights.39-0.01.h5')

    count = len(os.listdir("../../test/images/input/"))

    elem = 0
    IoU = np.zeros((count, 10), dtype=np.float32)

    for filename in os.listdir("../../test/images/input/"):

        image_inp = Image.open("../../test/images/input/" + filename)
        image_inp = np.array(image_inp)/255.
        inp = np.expand_dims(image_inp, axis=0)

        mask_name = "mask" + filename[4:]
        mask_inp = Image.open("../../test/masks/output/" + mask_name).convert("L")
        mask_inp = np.array(mask_inp)/255.

        y_pred = model.predict(inp, verbose=0)
        y_pred = np.reshape(y_pred, (240, 320))

        thresh = np.arange(0.0, 1.0, 0.1)

        
        for index, threshold in enumerate(thresh):

            y = y_pred.copy()
            y[y < threshold] = 0
            y[y >= threshold] = 1


            union = []
            inter = []

            for i in range(240):
                for j in range(320):
                    if mask_inp[i][j] == 1 or y[i][j] == 1:
                        union.append(1)
                    if mask_inp[i][j] == 1 and y[i][j] == 1:
                        inter.append(1)

            IoU[elem][index] = (len(inter)/(len(union)))
        
        elem += 1

        np.save("IoU.txt", IoU)

# make_prediction()