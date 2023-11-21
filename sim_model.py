import torch
import sys
import os

model_path = '/home/lastbasket/Code/rm23/LCR_sjtu/yolov5/RM_120.pt'
model = torch.load(model_path)

print(model.keys())