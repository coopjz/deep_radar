import os
import random
import shutil
import numpy as np
from os import listdir
from PIL import Image
from tqdm import tqdm

# set the path to the dataset
dataset_1_path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/combined/temp/'
dataset_2_path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/armor_dataset_v4/'
# set the path to the train val test folder
train_path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/combined/train/'
val_path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/combined/val/'
test_path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/combined/test/'

# set the percentage of train val test
train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

# get the list of all the images
image_list_1 = []
image_list_2 = []

# rename images function
def rename(path,namelabel):
    for filename in listdir(path):
        os.rename(path+filename,path+namelabel+filename)
        

if __name__ == '__main__':
    rename(dataset_1_path+'images/','field_')
    rename(dataset_1_path+'labels/','field_')

