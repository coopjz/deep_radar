import os
import random
import shutil
import numpy as np
from os import listdir
from PIL import Image
from tqdm import tqdm

# set the path to the dataset
path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/combined/images/'
label_path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/combined/labels/'

#change label function, only change the label start with 'field_'
def change_label_number(label_path,number=12):
    for filename in listdir(label_path):
        if filename.startswith('field_'):
            f = open(label_path + filename, 'r')
            lines = f.readlines()
            f.close()
            f = open(label_path + filename, 'w')
            for line in lines:
                line = line.split(' ')
                line[0] = str(number)
                line = ' '.join(line)
                f.write(line)
            f.close()

print('change label number:')
change_label_number(label_path,12)
print('change label number done')
