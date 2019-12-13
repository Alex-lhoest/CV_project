import pandas as pd
import numpy as np
import os
from shutil import copyfile
import xml.etree.ElementTree as et
import re

import sys

def get_annotations(write=False):

    count_road = 0
    count_sudoku = 0
    count_soccer = 0

    count_elps_eye = 0
    count_elps_soccer = 0
    prev_name = ""

    path = "../CV2019_Project/CV2019_Annots_ElpsSoccer.csv"
    save_path = "../annotations/"

    f = pd.read_csv(path, header=None, delimiter=';', na_filter=False)

    for idx, row in f.iterrows():
        element = re.sub("[^A-Z]", "", row[2])
        # Line element with 2 points only
        if element == "LINESTRING":
            if not row[6]:
                name = row[0]
                if name != prev_name:
                    if name.find("sudoku") >= 0:
                        count_sudoku += 1
                        suffix = "sudoku_" + str(count_sudoku) + ".png"
                    elif name.find("soccer") >= 0:
                        count_soccer += 1
                        suffix = "soccer_" + str(count_soccer) + ".png"
                    else:
                        count_road += 1
                        suffix = "road_" + str(count_road) + ".png"

                    if write:
                        copyfile("../Temp/" + name, "../images/" + suffix)

                filename = suffix[:-4]
                
                x1 = row[2]
                y1 = row[3]
                x2 = row[4]
                y2 = row[5]
                coord = (np.array((x1, y1, x2, y2))).reshape([1, 4])
                df = pd.DataFrame(coord)
                if write:
                    df.to_csv(save_path + str(filename) + ".txt", mode='a', header=None, index=False)
                prev_name = name
        
        elif element == "POLYGON":
            i = 3
            coord = []
            name = row[0]
            # print(idx)
            if name != prev_name:
                if name.find('eye') >= 0:
                    count_elps_eye += 1
                    suffix = "elps_eye_" + str(count_elps_eye) + ".png"
                elif name.find('soccer') >= 0:
                    count_elps_soccer += 1
                    suffix = "elps_soccer_" + str(count_elps_soccer) + ".png"

                if write:
                    copyfile("../Temp/" + name, "../images/" + suffix)
            
            filename = suffix[:-4]
            while row[i]:
                coord.append(row[i])
                i += 1
            coord = (np.array(coord)).reshape([1, len(coord)])
            df = pd.DataFrame(coord)
            if write:
                df.to_csv(save_path + str(filename) + ".txt", mode='a', header=None, index=False)
            prev_name = name
               
      
def change_dir():
    for fol_name in os.listdir("../images/"):
        if fol_name[0] != '.':
            for sub_name in os.listdir("../images/" + fol_name):
                if sub_name[0] != '.':
                    for name in os.listdir("../images/" + fol_name + "/" + sub_name):
                        copyfile("../images/" + fol_name + "/" + sub_name + "/" + name, '../Temp/' + name)


get_annotations()
# change_dir()