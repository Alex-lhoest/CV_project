import sys
import os
import numpy as np
from statistics import mean

def IOU_soccer():
    list_IOU = list()
    nb_real_ellipses = 0
    nb_found_ellipses = 0
    bad_files = list()
    for file in os.listdir("positions"):
        fp1 = open("positions/{}".format(file), "r")
        fp2 = open("Coord_test/{}".format(file), "r")
        
        lines1 = fp1.readlines()
        nb_lines1 = np.shape(lines1)
        lines2 = fp2.readlines()
        nb_lines2 = np.shape(lines2)
        
        nb_elps1 = nb_lines1[0]
        nb_elps2 = nb_lines2[0]
        if nb_elps1 != nb_elps2:
            bad_files.append(file)
        nb_real_ellipses += nb_elps2
        nb_found_ellipses += nb_elps1
        
        for i in range(nb_elps1):
            line1= lines1[i].split(' ')
            if line1 == ['\n']:
                break
            xmin1 = int(line1[0])
            ymin1= int(line1[1])
            xmax1 = int(line1[2])
            ymax1 = int(line1[3])
            
            length1 = (xmax1 -xmin1)
            height1 = (ymax1 - ymin1)
            area1 = length1*height1
            intersection = 0
            
            IOUs = list()
            for  j in range(nb_elps2):
                line2= lines2[j].split(' ')
                xmin2 = int(line2[0])
                ymin2 = int(line2[1])
                xmax2 = int(line2[2])
                ymax2 = int(line2[3])
            
                length2 = (xmax1 -xmin1)
                height2 = (ymax1 - ymin1)
                area2 = length2 * height2
                interminx = max(xmin1, xmin2)
                intermaxx = min(xmax1, xmax2)
                interminy = max(ymin1, ymin2)
                intermaxy = min(ymax1, ymax2)
                
                if (intermaxx - interminx) > 0 and (intermaxy - interminy) > 0:
                    intersection = (intermaxx - interminx) * (intermaxy - interminy)
                else :
                    intersection = 0
                
                union = area1 + area2 - intersection
                IOUs.append(intersection / union)
            IOU = max(IOUs)
            if IOU > 0:
                list_IOU.append(IOU)
                
    return nb_real_ellipses, nb_found_ellipses, bad_files, mean(list_IOU), list_IOU

#print("nb_real_ellipses: {}".format(nb_real_ellipses))
#print("nb_found_ellipses: {}".format(nb_found_ellipses))
#print("bad files : {}".format(bad_files))
#print("mean IOU: {}".format(mean(list_IOU)))
