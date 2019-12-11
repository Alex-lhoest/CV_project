import shutil
import os
import random

def sort():
    for filename in os.listdir('../images'):
        if filename.find('elps') > 0:
            shutil.move('../images/' + filename, '../train_soccer/' + filename)
    for filename in os.listdir('../bounding_boxes'):
        shutil.move('../bounding_boxes'+filename, '../train_soccer/' +filename)

def split():
    count = len(os.listdir('../train_soccer/images'))
    split = int(count*0.2)
    rnd = random.sample(range(count), split)

    for i in rnd:
        shutil.move('../train_soccer/elps_soccer_' + str(i+1) + '.png', '../test_soccer/elps_soccer_' + str(i+1) + '.png')
        shutil.move('../train_soccer/elps_soccer_' + str(i+1) + '.txt', '../test_soccer/elps_soccer_' + str(i+1) + '.txt')

# sort()
# split()
