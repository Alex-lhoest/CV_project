from unet import *
from keras.preprocessing import image
from keras.callbacks import TensorBoard
import numpy as np
import keras
import time

scaling = 1/255.
target_size = (240, 320)
batch_size = 2

# Define the generator used for the training
def train_generator():

    # Scaling the input between [0 1]
    image_datagen = image.ImageDataGenerator(rescale=scaling)
    mask_datagen = image.ImageDataGenerator(rescale=scaling)
    
    # Retrieve the Images and The masks according to their directories
    # Batch size is 2
    image_generator = image_datagen.flow_from_directory(
        '../train/images', target_size=target_size, batch_size=batch_size,
        class_mode=None, shuffle=False)
    
    mask_generator = mask_datagen.flow_from_directory(
        '../train/masks', color_mode='grayscale', target_size=target_size, 
        batch_size=batch_size, class_mode=None, shuffle=False)
    
    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)
    
    return train_generator 

# Define the generator used for the validation
def test_generator():
    
    # Scaling the input between [0 1]
    image_test_datagen = image.ImageDataGenerator(rescale=scaling)
    mask_test_datagen = image.ImageDataGenerator(rescale=scaling)

    # Retrieve the Images and The masks according to their directories
    # Batch size is 2
    image_test_generator = image_test_datagen.flow_from_directory(
        '../eval/images', target_size=target_size, batch_size=batch_size,
        class_mode=None, shuffle=False)
    
    mask_test_generator = mask_test_datagen.flow_from_directory(
        '../eval/masks', color_mode='grayscale', target_size=target_size, 
    batch_size=batch_size, class_mode=None, shuffle=False)
    
    # combine generators into one which yields image and masks
    test_generator = zip(image_test_generator, mask_test_generator)
    
    return test_generator

train_generator = train_generator()
test_generator = test_generator()

# Create the model with the unet architecture
model = unet()

history = keras.callbacks.History()

# Callbacks used to save the model if the model improve on the validation loss

save_path = '../models/300epochs/'

if not os.path.isdir(save_path):
    os.makedirs(save_path)

checkpoint = keras.callbacks.ModelCheckpoint(save_path + 'unet_weights.{epoch:02d}-{val_loss:.2f}.h5', 
                                             verbose=1, 
                                             monitor='val_loss', save_best_only=True, 
                                             mode='auto')

# Tensorboard to check the evolution of the training                                      
NAME = unet
tensorboard = TensorBoard(log_dir=save_path + 'logs/{}'.format(NAME))
                                       

# Fit the model with our model with the generators
model.fit_generator(train_generator, steps_per_epoch=1050, epochs=300,
                    shuffle=False, validation_data=test_generator, validation_steps=525//2,
                    callbacks=[checkpoint, history, tensorboard])
