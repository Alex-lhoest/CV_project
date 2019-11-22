import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

#############################################################################

def multiPlot( n, m, img_tuple, title_tuple, cmap_tuple=None, dispType_tuple=None, vmin_tuple=None, vmax_tuple=None):
    plt.figure(figsize=(20,10))
    for i in np.arange( n * m):
        if img_tuple[i] is None:
            continue

        if cmap_tuple is not None:
            cmap = cmap_tuple[i]
        else:
            cmap=None

        if dispType_tuple is not None:
            dispType = dispType_tuple[i]
        else:
            dispType=None

        if vmin_tuple is not None:
            vmin = vmin_tuple[i]
        elif dispType == 'histogram':
            vmin = 0
        else:
            vmin=None

        if vmax_tuple is not None:
            vmax = vmax_tuple[i]
        elif dispType == 'histogram':
            vmax = 255
        else:
            vmax=None

        if dispType == 'histogram':
            plt.subplot( n, m, i + 1), plt.hist( img_tuple[i].ravel(), bins=256, range=( vmin, vmax)) #, fc='k', ec='k')
        else:
            plt.subplot( n, m, i + 1), plt.imshow( img_tuple[i], cmap=cmap, vmin=vmin, vmax=vmax)
            plt.xticks([]), plt.yticks([])

        plt.title( title_tuple[i])

    #manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())
    plt.show()

#############################################################################

def saturate_cast_uint8( image):
    return np.where( image > 255.0, 255.0, np.where( image < 0.0, 0.0, image)).astype( np.uint8)

#############################################################################

from matplotlib.colors import LinearSegmentedColormap

def getRandomColorMap( num_colors):
    colors = np.random.rand( num_colors, 3) * 0.75
    colors[0, :] = 1
    colors = tuple(map(tuple, colors))

    labelColorMap = LinearSegmentedColormap.from_list('labelColorMap', colors, N=num_colors)

    return labelColorMap

