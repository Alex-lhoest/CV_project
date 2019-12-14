import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import tools
import os
import sys


def get_ellipse_param(img, conn_comp):
    num_labels = conn_comp[0]   # The first cell is the number of labels
    img_labels = conn_comp[1]   # The second cell is the label matrix
    stats = conn_comp[2]        # The third cell is the stat matrix
    centroids = conn_comp[3].astype(int)    # The fourth cell is the centroid matrix
    
    width = img.shape[1]
    height = img.shape[0]
    
    x_range = np.arange(width)
    y_range = np.arange(height)
    x_array, y_array = np.meshgrid(x_range, y_range)
    
    a=0
    b=0
    xc=0
    yc=0
    theta=0
    
    # Draw ellipses
    img_elps = cv2.cvtColor( img, cv2.COLOR_GRAY2RGB)
    for label in range(1, num_labels):
    #for label in range(1,2):
        xc = centroids[label,0]
        yc = centroids[label,1]
        xl = x_array[img_labels == label] - xc
        yl = y_array[img_labels == label] - yc
        n = np.sum(img_labels == label)
        Mxx = np.sum(np.square(xl)) / n
        Myy = np.sum(np.square(yl)) / n
        Mxy = np.sum(xl * yl) / n
        discr = math.sqrt(4.0 * Mxy * Mxy + (Mxx - Myy) ** 2)
        a = int( math.sqrt( 2.0 * ( Mxx + Myy + discr)))
        b = int( math.sqrt( 2.0 * ( Mxx + Myy - discr)))
        theta = int( 0.5 * math.atan2( 2.0 * Mxy, Mxx - Myy) * 180.0 / np.pi)
        #print( 'Label %d: xc = %d, yc = %d, a = %d, b = %d, theta = %d' % ( label, xc, yc, a, b, theta))
        
        cv2.ellipse( img_elps, (xc, yc), (a, b), theta, 0, 360, (255,0,0), thickness=1)
        cv2.circle( img_elps, (xc, yc), 1, (255,0,0), thickness=2)

    return img_elps, a, b, xc, yc, theta



def getElpsParameters(input_arr):

    output_arr = np.array(input_arr)
    array = output_arr.astype(np.uint8)
    array = array*255

    ## Get ellipse parameters + draw
    connectivity = 4
    ret,thresh1 = cv2.threshold(array,100,255,cv2.THRESH_BINARY)
    conn_comp = cv2.connectedComponentsWithStats(thresh1, connectivity, cv2.CV_32S)

    img_elps, a, b, xc, yc, theta = get_ellipse_param(input_arr, conn_comp)

    return a, b, xc, yc, theta



