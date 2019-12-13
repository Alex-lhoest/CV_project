import shutil
import os
import time
import sys

tf = open("../all_im/train.txt", "w")

vf = open ("../all_im/val.txt", "w")

testf = open ("../all_im/test.txt", "w")

with open ("../all_im/all_im.txt") as fp:
    for line in fp:
        line = line.split('/')[-1]
        if line[:-1] in os.listdir('../soccer_sets/training_set/img'):
            tf.write(line)
        if line[:-1] in os.listdir('../soccer_sets/validation_set/img'):
            vf.write(line)
        if line[:-1] in os.listdir('../soccer_sets/test_set/img'):
            testf.write(line)



tf.close()
vf.close()
testf.close()
