
import os
from os import listdir
from PIL import Image


for filename in listdir('/home/cooper/Documents/RM2023-Deep-Radar/dataset/armor_dataset_v4/images/'):
    if filename.endswith('.jpg'):
        try:
            img = Image.open('/home/cooper/Documents/RM2023-Deep-Radar/dataset/armor_dataset_v4/images/' +
                             filename)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            print(filename)
            os.remove(
                '/home/cooper/Documents/RM2023-Deep-Radar/dataset/armor_dataset_v4/images/'+filename)
