from keras.models import load_model
import numpy as np
from PIL import Image
import keras
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from unet import mean_iou
import sys
from getElpsParam import getElpsParameters

dependencies = {
    'mean_iou': mean_iou
}

def make_prediction():

    # Load the models
    # model = load_model('../../models/final_model/unet_weights.39-0.01.h5')
    model = load_model('unet_weights.39-0.01.h5')

    # Load the image
    image_inp = Image.open('../../test/images/input/elps_eye_2204.png')
    # image_inp = Image.open('../../NoEllipses/noelps_eye2015-01-27_02-27-30-001.png')

    mask_inp = Image.open('../../test/masks/output/mask_eye_2204.png')
    mask = np.array(mask_inp)/255.

    # Scale the input between [0 1] and change the arrays (1, 240, 320, 3)
    image_inp = np.array(image_inp)/255.
    inp = np.expand_dims(image_inp, axis=0)
    
    # Predict the segmanted mask given the model
    y_pred = model.predict(inp, verbose=0)

    # Reshape the output (240, 320)
    y_pred = np.reshape(y_pred, (240, 320))

    threshold = 0.7
    y_pred[y_pred < threshold] = 0
    y_pred[y_pred >= threshold] = 1

    a, b, x, y, theta = getElpsParameters(y_pred)

    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for i in range(240):
        for j in range(320):
            # True Positive in green
            if y_pred[i][j] and mask[i][j][0]:
                image_inp[i][j][1] = 1
                TP += 1
            # False Positive in red
            if y_pred[i][j] and not mask[i][j][0]:
                image_inp[i][j][0] = 1
                FP += 1
            # False Negative in blue
            if not y_pred[i][j] and mask[i][j][0]:
                image_inp[i][j][2] = 1
                FN += 1
            if not y_pred[i][j] and not mask[i][j][0]:
                TN += 1

    if TP != 0 and TN != 0: 
        prec = TP/(TP + FP)
    else:
        prec = 0
    rec = TP/(TP + FN)
    mr = FN/(FN + TP)
    acc = (TP+TN)/(TP+TN+FP+FN)  

    print("Precision : {}, Recall : {}, Miss Rate : {}, Accuracy : {}".format(prec, rec, mr, acc))

    print("a : {}, b : {}, x : {}, y : {}, theta : {}".format(a, b, x, y, theta))
    # image_inp[..., 0] += y_pred
    # image_inp[image_inp > 1] = 1

    # Transform the array in a image representation
    image_out = keras.preprocessing.image.array_to_img(image_inp)
    image_out.show()
    mask_inp.show()

    # Save the image
    # image_out.save("test.png", format='png')

def threshold_results(file=None):

    if not file:

        model = load_model('../../models/final_model/unet_weights.39-0.01.h5')

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

    if file:
        IoU = np.load(file)

    mean = np.mean(IoU, axis=0)
    median = np.median(IoU, axis=0)

    print("Mean of the IoU for different threshold value : {}".format(mean))
    print("Median of the IoU for different threshold value : {}".format(median))

def metrics_results(file=None, threshold=0.7):

    if not file:
        # model = load_model('../../models/final_model/unet_weights.39-0.01.h5')
        model = load_model('unet_weights.39-0.01.h5')

        count = len(os.listdir("../../test/images/input/"))

        elem = 0

        precision = np.zeros((count, 1), dtype=np.float32)
        recall = np.zeros((count, 1), dtype=np.float32)
        miss_rate = np.zeros((count, 1), dtype=np.float32)
        accuracy = np.zeros((count, 1), dtype=np.float32)

        for filename in os.listdir("../../test/images/input/"):

            image_inp = Image.open("../../test/images/input/" + filename)
            image_inp = np.array(image_inp)/255.
            inp = np.expand_dims(image_inp, axis=0)

            mask_name = "mask" + filename[4:]
            mask_inp = Image.open("../../test/masks/output/" + mask_name)
            mask = np.array(mask_inp)/255.

            y_pred = model.predict(inp, verbose=0)
            y_pred = np.reshape(y_pred, (240, 320))

            threshold = threshold
            y_pred[y_pred < threshold] = 0
            y_pred[y_pred >= threshold] = 1

            TP = 0
            FP = 0
            FN = 0
            TN = 0

            for i in range(240):
                for j in range(320):
                    # True Positive in green
                    if y_pred[i][j] and mask[i][j][0]:
                        TP += 1
                    # False Positive in red
                    if y_pred[i][j] and not mask[i][j][0]:
                        FP += 1
                    # False Negative in blue
                    if not y_pred[i][j] and mask[i][j][0]:
                        FN += 1
                    # True Negative
                    if not y_pred[i][j] and not mask[i][j][0]:
                        TN += 1

            if TP != 0 and TN != 0: 
                prec = TP/(TP + FP)
            else:
                prec = 0
            rec = TP/(TP + FN)
            mr = FN/(FN + TP)
            acc = (TP+TN)/(TP+TN+FP+FN)    

            precision[elem] = prec
            recall[elem] = rec
            miss_rate[elem] = mr
            accuracy[elem] = acc
            
            elem += 1

        np.save("precision_{}".format(threshold), precision)
        np.save("recall_{}".format(threshold), recall)
        np.save("miss_rate_{}".format(threshold), miss_rate)
        np.save("accuracy_{}".format(threshold), accuracy)
    
    if file:
        precision = np.load(file[0])
        recall = np.load(file[1])
        miss_rate = np.load(file[2])
        accuracy = np.load(file[3])

    mean_p = np.mean(precision, axis=0)
    median_p = np.median(precision, axis=0)

    print("Mean of the precision : {}".format(mean_p))
    print("Median of the precision : {}".format(median_p))

    mean_r = np.mean(recall, axis=0)
    median_r = np.median(recall, axis=0)

    print("Mean of the recall : {}".format(mean_r))
    print("Median of the recall : {}".format(median_r))

    mean_m = np.mean(miss_rate, axis=0)
    median_m = np.median(miss_rate, axis=0)

    print("Mean of the miss rate : {}".format(mean_m))
    print("Median of the miss rate : {}".format(median_m))

    mean_a = np.mean(accuracy, axis=0)
    median_a = np.median(accuracy, axis=0)

    print("Mean of the accuracy : {}".format(mean_a))
    print("Median of the accuracy : {}".format(median_a))


# make_prediction()

# threshold_results(file="IoU.npy")

# metrics_results(threshold=0.9)
