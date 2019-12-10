import shutil
import os
import random
import time
import sys

def sort():
    for filename in os.listdir('../images'):
        if filename.find('eye') > 0:
            shutil.move('../images/' + filename, '../train/' + filename)

def split():

    index_eval = []
    for filename in sorted(os.listdir('../train/images/input')):
        idx = filename[9:-4]
        index_eval.append(idx)


    split_eval = int(len(index_eval)*0.2)
    rnd_eval = random.sample(index_eval, split_eval)

    for i in rnd_eval:
        shutil.move('../train/images/input/elps_eye_' + i + '.png', '../eval/images/input/elps_eye_' + i + '.png')
        shutil.move('../train/masks/output/mask_eye_' + i + '.png', '../eval/masks/output/mask_eye_' + i + '.png')

    index_test = []
    for filename in os.listdir('../train/images/input'):
        idx = filename[9:-4]
        index_test.append(idx)

    split_test = int(2625*0.1)
    rnd_test = random.sample(index_test, split_test)

    for i in rnd_test:
        shutil.move('../train/images/input/elps_eye_' + i + '.png', '../test/images/input/elps_eye_' + i + '.png')
        shutil.move('../train/masks/output/mask_eye_' + i + '.png', '../test/masks/output/mask_eye_' + i + '.png')

# sort()
split()