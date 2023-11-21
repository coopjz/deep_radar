# the program helps split the dataset into train val and test set

import os
import random
import shutil
import numpy as np
from os import listdir
from PIL import Image
from tqdm import tqdm

# set the path to the dataset
path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/armor_dataset_v4/images/'
label_path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/armor_dataset_v4/labels/'
# set the path to the train val test folder
train_path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/armor_dataset_v4/train/'
val_path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/armor_dataset_v4/val/'
test_path = '/home/cooper/Documents/RM2023-Deep-Radar/dataset/armor_dataset_v4/test/'

# set the percentage of train val test
train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

# get the list of all the images
image_list = []


for filename in listdir(path):
    if filename.endswith('.jpg'):
        try:
            img = Image.open(path + filename)  # open the image file
            img.verify()  # verify that it is, in fact an image
            image_list.append(filename)
        except (IOError, SyntaxError) as e:
            print(filename)

# shuffle the image list
random.shuffle(image_list)

# get the number of images
num_images = len(image_list)

# get the number of train val test
num_train = int(train_percent * num_images)
num_val = int(val_percent * num_images)

# split the dataset
train_list = image_list[:num_train]
val_list = image_list[num_train:num_train + num_val]
test_list = image_list[num_train + num_val:]

# check if the train val test folder exists
if not os.path.exists(train_path):
    os.makedirs(train_path)
    os.makedirs(train_path + 'images/')
    os.makedirs(train_path + 'labels/')
if not os.path.exists(val_path):
    os.makedirs(val_path)
    os.makedirs(val_path + 'images/')
    os.makedirs(val_path + 'labels/')
if not os.path.exists(test_path):
    os.makedirs(test_path)
    os.makedirs(test_path + 'images/')
    os.makedirs(test_path + 'labels/')

# copy the images and labels to the train val test folder
for filename in tqdm(train_list):
    try:
        txt = open(label_path + filename[:-4] + '.txt', 'r')
        txt.close()
    except FileNotFoundError:
        print(filename)
        continue
    shutil.copy(path + filename, train_path+'images/' + filename)
    shutil.copy(label_path + filename[:-4] + '.txt',
                train_path + 'labels/' + filename[:-4] + '.txt')

for filename in tqdm(val_list):
    try:
        txt = open(label_path + filename[:-4] + '.txt', 'r')
        txt.close()
    except FileNotFoundError:
        print(filename)
        continue
    shutil.copy(path + filename, val_path + 'images/' + filename)
    shutil.copy(label_path + filename[:-4] + '.txt',
                val_path + 'labels/' + filename[:-4] + '.txt')

for filename in tqdm(test_list):
    try:
        txt = open(label_path + filename[:-4] + '.txt', 'r')
        txt.close()
    except FileNotFoundError:
        print(filename)
        continue
    shutil.copy(path + filename, test_path + 'images/' + filename)
    shutil.copy(label_path + filename[:-4] + '.txt',
                test_path + 'labels/' + filename[:-4] + '.txt')

# print the number of train val test
print('Number of train set: {}'.format(len(train_list)))
print('Number of val set: {}'.format(len(val_list)))
print('Number of test set: {}'.format(len(test_list)))
