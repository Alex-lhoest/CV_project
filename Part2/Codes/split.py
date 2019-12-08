import shutil
import os
import random

def sort():
    for filename in os.listdir('../images'):
        if filename.find('eye') > 0:
            shutil.move('../images/' + filename, '../train/' + filename)

def split():
    count = len(os.listdir('../train/images'))
    split = int(count*0.2)
    rnd = random.sample(range(count), split)

    for i in rnd:
        shutil.move('../train/images/elps_eye_' + str(i+1) + '.png', '../eval/images/elps_eye_' + str(i+1) + '.png')
        shutil.move('../train/masks/mask_eye_' + str(i+1) + '.png', '../eval/masks/mask_eye_' + str(i+1) + '.png')

# sort()
# split()