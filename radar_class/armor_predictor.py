import torch
import numpy as np
from pathlib import Path
import sys
import platform
import os
import argparse
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
sys.path.append('/home/cooper/Documents/RM2023-Deep-Radar/yolov5')

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, cv2,
                                  increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                        letterbox, mixup, random_perspective)



# class YoloV5(object):
#     def __init__(self):
#         '''
#         对于示例类，不会提供神经网络预测功能，但对于我们提供的demo，可以加载pkl来获得实际的预测结果

#         :param weights:pkl文件的存放地址
#         '''
#         self.device = torch.device(
#             "cuda" if torch.cuda.is_available() else "cpu")
#         self.model = DetectMultiBackend("/home/cooper/Documents/RM2023-Deep-Radar/RM_250.onnx",
#                                         device=self.device, data='yolov5/data/sjtu.yaml')
#         self.num_net = cv2.dnn.readNetFromONNX('/home/cooper/Documents/RM2023-Deep-Radar/mlp.onnx')
#         # self.num_net = DetectMultiBackend(
#         #     weights='yolov5/armor.pt', device=self.device, data='yolov5/data/armor.yaml')
#         print('init yolov5')
#         self.pred_dict = {0: 'car_red',
#                           1: 'car_blue',
#                           2: 'car_unknow',
#                           3: 'watcher_red',
#                           4: 'watcher_blue',
#                           5: 'watcher_unknow',
#                           6: 'armor_red',
#                           7: 'armor_blue',
#                           8: 'armor_grey'}
#         self.armor_num_list = [1, 2, 3, 4, 5,
#                                'outpost', 'sentry', 'base', 'negative']
#         self.car_armor_map = {'car_red': [
#             8, 9, 10, 11, 12], 'car_blue': [1, 2, 3, 4, 5]}

#     def infer(self, imgs):
#         '''
#         这个函数用来预测
#         :param imgs:list of input images

#         :return:
#         img_preds: 车辆预测框，a list of the prediction to each image 各元素格式为(predicted_class,conf_score,bounding box(format:x0,y0,x1,y1))

#         [#图片级列表start
#             [#图片1start
#                 ['car_blue_2', 0.8359227180480957, [2330.0, 1371.0, 2590.0, 1617.0]] # 每项预测
#             ]#图片1end
#             , 
#             #图片2start
#             []
#             #图片2end
#         ], #图片级列表end

#         list[cars_per_img(list)[], ]

#         car_locations: 对于每张图片装甲板预测框（车辆定位） np.ndarray 和对应的车辆预测框(与装甲板预测框的车辆预测框序号对应）的列表
#         上述两个成员具体定义为：
#         （1）装甲板预测框格式,(N,装甲板四点+装甲板网络置信度+装甲板类型+其对应的车辆预测框序号（即其为哪个车辆预测框ROI区域预测生成的）+四点的bounding box)
#         其他敌方提到该格式，会写为（N,fp+conf+cls+img_no+bbox)

#         （2）车辆预测框格式 np.ndarray (N,x0+y0+x1+y1)

#         armor plate: left up, left down, right up, right down 4 points (0,8), acc 8, num 9, 
#         corresponding car in img_preds 10,
#         bbox: (x1, y1, w, h) (11, 15)

#         list[per_img_pred(np array), per_img_pred(np array), ...]

#         [#图片级列表start
#             #图片1
#             [
#             # numpy数组（N*15） N为该图片中预测出的装甲板数
#             array([[2.4680000e+03, 1.5520000e+03, 2.4690000e+03, 1.5720000e+03,
#                     2.5050000e+03, 1.5700000e+03, 2.5040000e+03, 1.5510000e+03,
#                     9.5630890e-01, 2.0000000e+00, 0.0000000e+00, 2.4680000e+03,
#                     1.5510000e+03, 3.7000000e+01, 2.1000000e+01],
#                 [2.3710000e+03, 1.5190000e+03, 2.3660000e+03, 1.5350000e+03,
#                     2.3850000e+03, 1.5450000e+03, 2.3890000e+03, 1.5280000e+03,
#                     9.2683524e-01, 2.0000000e+00, 0.0000000e+00, 2.3660000e+03,
#                     1.5190000e+03, 2.3000000e+01, 2.6000000e+01]], dtype=float32)
#                     , 
#                     # numpy数组（M*4） M为该图片中预测出的车辆（car，所以watcher和base不算）数，和上一个对应
#                     array([[2330., 1371., 2590., 1617.]], dtype=float32)], 
#                     # 图片2 (没有任何装甲板预测为两个None)
#                     [None, None]
#         ]# 图片级列表end

#         location: [[None, None], [None, None]]
#         results: [[['car_blue_2', 0.7970007061958313, [2169.0, 1700.0, 2507.0, 1977.0]]], []]
#         location: [array([[       2288,        1867,        2288,        1939,        2373,
#                                 1867,        2373,        1939,     0.70498,           0,
#                                 2288,        1867,          85,          72]], dtype=float32), [None, None]]

#         example from real yolov5:

#         [[['car_red_2', 0.8677165508270264, [674.0, 1558.0, 1943.0, 2985.0]]]] 
#             [array([[       1409,        2567,        1409,        2762,        
#             1639,        2567,        1639,        2762,     
#             0.83465,           0,        1409,        2567,         
#             230,         195],
#             [        855,        2623,         855,        2829,        
#             1037,        2623,        1037,        2829,     
#             0.79812,           0,         855,        2623,         
#             182,         206]], dtype=float32)]

#         tensor([[1.92200e+03, 1.15200e+03, 2.14600e+03, 1.31300e+03, 8.04348e-01, 1.00000e+00],
#         [1.97700e+03, 1.25100e+03, 2.04300e+03, 1.29900e+03, 7.29248e-01, 7.00000e+00],
#         [2.07200e+03, 1.23700e+03, 2.12300e+03, 1.28700e+03, 3.68224e-01, 7.00000e+00]], device='cuda:0')

#         '''
#         img_preds, car_locations = [], []
#         ori_shape = (imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2])

#         preds = []
#         for img in imgs:
#             # in format of H, W, C and BGR
#             # copy im0 -> letterbox -> transform -> as_continuous
#             # -> tensor -> .float() -> /= 255 -> im = im[None]

#             img0 = img.copy()

#             # gray img: BGR -> gray -> /255 -> float32
#             img0_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) /
#                          255.0).astype(np.float32)

#             img = letterbox(img0, [640, 640], stride=32, auto=False)[
#                 0]  # padded resize
#             img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#             img = np.ascontiguousarray(img)  # contiguous
#             img = torch.from_numpy(img).to(self.device).float()
#             img = img / 255
#             if len(img.shape) == 3:
#                 img = img[None]  # expand for batch dim

#             # torch.Size([1, 3, 448, 640])
#             pred = self.model(img)
#             # output of the pred: shape (b, N, 6)
#             # x1, y1, x2, y2, acc, class

#             pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

#             pred[0][:, :4] = scale_boxes(
#                 img.shape[2:], pred[0][:, :4], ori_shape).round()
#             car_pred_per_img = []
#             armor_pred_per_img = []
#             car_bbox_pre_img = []

#             ori_car_pred_per_img = []
#             ori_armor_pred_per_img = []
#             armor_num_per_img = []
#             if pred[0].shape[0] == 0:
#                 img_preds.append([])
#                 car_locations.append([None, None])
#                 continue

#             for i in range(pred[0].shape[0]):
#                 if (pred[0][i][5] == 0) or (pred[0][i][5] == 1):
#                     ori_car_pred_per_img.append(pred[0][i])
#                 elif (pred[0][i][5] == 6) or (pred[0][i][5] == 7):
#                     armor_x1 = pred[0][i][0].cpu().numpy().astype(np.uint16)
#                     armor_y1 = pred[0][i][1].cpu().numpy().astype(np.uint16)
#                     armor_x2 = pred[0][i][2].cpu().numpy().astype(np.uint16)
#                     armor_y2 = pred[0][i][3].cpu().numpy().astype(np.uint16)
#                     armor_acc = pred[0][i][4].cpu().numpy().astype(np.uint16)

#                     # predict the number of armor
#                     # crop as x1+1/4*width, x2-1/4*width for the armor_area to remove the LEDs for armor
#                     armor_width = armor_x2 - armor_x1
#                     new_armor_x1 = armor_x1 + int(armor_width * 0.38)
#                     new_armor_x2 = armor_x2 - int(armor_width * 0.38)

#                     armor_area = img0_gray[armor_y1:armor_y2,
#                                            new_armor_x1:new_armor_x2]

#                     ori_armor_area = img0[armor_y1:armor_y2,
#                                           new_armor_x1:new_armor_x2]

#                     cv2.imwrite('test_armor.png', armor_area*255)
#                     cv2.imwrite('test_ori_armor.png', ori_armor_area)

#                     # input is row: 20 col: 28 format: CV_8UC1
#                     armor_area = cv2.resize(armor_area, (20, 28))
#                     blob = cv2.dnn.blobFromImage(armor_area, size=(20, 28))
#                     self.num_net.setInput(blob)
#                     # self.num_net(blob)
#                     out = self.num_net.forward()[0]
#                     max_out = max(out)
#                     softmax_prob = np.exp(out - max_out)
#                     soft_sum = softmax_prob.sum()
#                     softmax_prob = softmax_prob / soft_sum
#                     softmax_prob = softmax_prob[:8]
#                     max_index = np.where(
#                         softmax_prob == softmax_prob.max())[0][0]
#                     if max_index > 4:
#                         continue
#                     else:
#                         armor_index = self.armor_num_list[max_index]
#                         armor_num_per_img.append(armor_index)
#                         ori_armor_pred_per_img.append(pred[0][i])

#             car_index = 0

#             for i in range(len(ori_car_pred_per_img)):
#                 car_x1 = ori_car_pred_per_img[i][0].cpu().numpy()
#                 car_y1 = ori_car_pred_per_img[i][1].cpu().numpy()
#                 car_x2 = ori_car_pred_per_img[i][2].cpu().numpy()
#                 car_y2 = ori_car_pred_per_img[i][3].cpu().numpy()
#                 car_acc = ori_car_pred_per_img[i][4].cpu().numpy()
#                 car_cls = int(ori_car_pred_per_img[i][5].cpu().numpy())
#                 car_num = -1

#                 for j in range(len(ori_armor_pred_per_img)):
#                     armor_x1 = ori_armor_pred_per_img[j][0].cpu().numpy()
#                     armor_y1 = ori_armor_pred_per_img[j][1].cpu().numpy()
#                     armor_x2 = ori_armor_pred_per_img[j][2].cpu().numpy()
#                     armor_y2 = ori_armor_pred_per_img[j][3].cpu().numpy()
#                     armor_acc = ori_armor_pred_per_img[j][4].cpu().numpy()

#                     # if the armor is in the range of the car then the car num is the armor num
#                     if (armor_x1 >= car_x1) and (armor_y1 >= car_y1) and (armor_x2 <= car_x2) and (armor_y2 <= car_y2):
#                         car_num = armor_num_per_img[j]
#                         single_armor_pred = [armor_x1, armor_y1,
#                                              armor_x1, armor_y2,
#                                              armor_x2, armor_y1,
#                                              armor_x2, armor_y2,
#                                              armor_acc, self.car_armor_map[self.pred_dict[car_cls]][car_num-1],
#                                              car_index,
#                                              armor_x1, armor_y1,
#                                              armor_x2-armor_x1, armor_y2-armor_y1]

#                         armor_pred_per_img.append(single_armor_pred)
#                 if car_num == -1:
#                     continue
#                 car_index = car_index + 1
#                 single_car_pred = [f'{self.pred_dict[car_cls]}_{car_num}', float(car_acc),
#                                    [float(car_x1), float(car_y1), float(car_x2), float(car_y2)]]
#                 car_bbox_pre_img.append([car_x1, car_y1, car_x2, car_y2])
#                 car_pred_per_img.append(single_car_pred)

#             if len(car_pred_per_img) == 0:
#                 img_preds.append([])
#                 car_locations.append([None, None])
#                 continue

#             img_preds.append(car_pred_per_img)
#             car_locations.append([np.array(armor_pred_per_img).astype(np.float32),
#                                  np.array(car_bbox_pre_img).astype(np.float32)])

#             preds.append(pred)

#         return img_preds, car_locations

class YoloV5(object):
    def __init__(self):
        '''
        对于示例类，不会提供神经网络预测功能，但对于我们提供的demo，可以加载pkl来获得实际的预测结果

        :param weights:pkl文件的存放地址
        '''
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = DetectMultiBackend("/home/cooper/Documents/RM2023-Deep-Radar/yolov5/runs/train/car_30011/weights/best.pt",
                                        device=self.device, data='yolov5/data/car.yaml')
        self.num_net = DetectMultiBackend('/home/cooper/Documents/RM2023-Deep-Radar/yolov5/armor.pt',device=self.device, data='yolov5/data/armor.yaml')
        # self.num_net = DetectMultiBackend(
        #     weights='yolov5/armor.pt', device=self.device, data='yolov5/data/armor.yaml')
        print('init yolov5')
        self.pred_dict = {0: 'car',
                          }
        self.armor_dict = {
            0:'B1',
            1:'B2',
            2:'B3',
            3:'B4',
            4:'B5',
            5:'B7',
            6:'R1',
            7:'R2',
            8:'R3',
            9:'R4',
            10:'R5',
            11:'R7',
        }
        self.armor_num_list = [1, 2, 3, 4, 5,
                               'outpost', 'sentry', 'base', 'negative']
        self.car_armor_map = {'car_red': [
            8, 9, 10, 11, 12], 'car_blue': [1, 2, 3, 4, 5]}

    def infer(self, imgs):
        '''
        这个函数用来预测
        :param imgs:list of input images

        :return:
        img_preds: 车辆预测框，a list of the prediction to each image 各元素格式为(predicted_class,conf_score,bounding box(format:x0,y0,x1,y1))

        [#图片级列表start
            [#图片1start
                ['car_blue_2', 0.8359227180480957, [2330.0, 1371.0, 2590.0, 1617.0]] # 每项预测
            ]#图片1end
            , 
            #图片2start
            []
            #图片2end
        ], #图片级列表end

        list[cars_per_img(list)[], ]

        car_locations: 对于每张图片装甲板预测框（车辆定位） np.ndarray 和对应的车辆预测框(与装甲板预测框的车辆预测框序号对应）的列表
        上述两个成员具体定义为：
        （1）装甲板预测框格式,(N,装甲板四点+装甲板网络置信度+装甲板类型+其对应的车辆预测框序号（即其为哪个车辆预测框ROI区域预测生成的）+四点的bounding box)
        其他敌方提到该格式，会写为（N,fp+conf+cls+img_no+bbox)

        （2）车辆预测框格式 np.ndarray (N,x0+y0+x1+y1)

        armor plate: left up, left down, right up, right down 4 points (0,8), acc 8, num 9, 
        corresponding car in img_preds 10,
        bbox: (x1, y1, w, h) (11, 15)

        list[per_img_pred(np array), per_img_pred(np array), ...]

        [#图片级列表start
            #图片1
            [
            # numpy数组（N*15） N为该图片中预测出的装甲板数
            array([[2.4680000e+03, 1.5520000e+03, 2.4690000e+03, 1.5720000e+03,
                    2.5050000e+03, 1.5700000e+03, 2.5040000e+03, 1.5510000e+03,
                    9.5630890e-01, 2.0000000e+00, 0.0000000e+00, 2.4680000e+03,
                    1.5510000e+03, 3.7000000e+01, 2.1000000e+01],
                [2.3710000e+03, 1.5190000e+03, 2.3660000e+03, 1.5350000e+03,
                    2.3850000e+03, 1.5450000e+03, 2.3890000e+03, 1.5280000e+03,
                    9.2683524e-01, 2.0000000e+00, 0.0000000e+00, 2.3660000e+03,
                    1.5190000e+03, 2.3000000e+01, 2.6000000e+01]], dtype=float32)
                    , 
                    # numpy数组（M*4） M为该图片中预测出的车辆（car，所以watcher和base不算）数，和上一个对应
                    array([[2330., 1371., 2590., 1617.]], dtype=float32)], 
                    # 图片2 (没有任何装甲板预测为两个None)
                    [None, None]
        ]# 图片级列表end

        location: [[None, None], [None, None]]
        results: [[['car_blue_2', 0.7970007061958313, [2169.0, 1700.0, 2507.0, 1977.0]]], []]
        location: [array([[       2288,        1867,        2288,        1939,        2373,
                                1867,        2373,        1939,     0.70498,           0,
                                2288,        1867,          85,          72]], dtype=float32), [None, None]]

        example from real yolov5:

        [[['car_red_2', 0.8677165508270264, [674.0, 1558.0, 1943.0, 2985.0]]]] 
            [array([[       1409,        2567,        1409,        2762,        
            1639,        2567,        1639,        2762,     
            0.83465,           0,        1409,        2567,         
            230,         195],
            [        855,        2623,         855,        2829,        
            1037,        2623,        1037,        2829,     
            0.79812,           0,         855,        2623,         
            182,         206]], dtype=float32)]

        tensor([[1.92200e+03, 1.15200e+03, 2.14600e+03, 1.31300e+03, 8.04348e-01, 1.00000e+00],
        [1.97700e+03, 1.25100e+03, 2.04300e+03, 1.29900e+03, 7.29248e-01, 7.00000e+00],
        [2.07200e+03, 1.23700e+03, 2.12300e+03, 1.28700e+03, 3.68224e-01, 7.00000e+00]], device='cuda:0')

        '''
        img_preds, car_locations = [], []
        ori_shape = (imgs[0].shape[0], imgs[0].shape[1], imgs[0].shape[2])

        preds = []
        for img in imgs:
            # in format of H, W, C and BGR
            # copy im0 -> letterbox -> transform -> as_continuous
            # -> tensor -> .float() -> /= 255 -> im = im[None]

            img0 = img.copy()

            # gray img: BGR -> gray -> /255 -> float32
            img0_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) /
                         255.0).astype(np.float32)

            img = letterbox(img0, [640, 640], stride=32, auto=False)[
                0]  # padded resize
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)  # contiguous
            img = torch.from_numpy(img).to(self.device).float()
            img = img / 255
            if len(img.shape) == 3:
                img = img[None]  # expand for batch dim

            # torch.Size([1, 3, 448, 640])
            pred = self.model(img)
            # output of the pred: shape (b, N, 6)
            # x1, y1, x2, y2, acc, class

            pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

            pred[0][:, :4] = scale_boxes(
                img.shape[2:], pred[0][:, :4], ori_shape).round()
            car_pred_per_img = []
            armor_pred_per_img = []
            car_bbox_pre_img = []

            ori_car_pred_per_img = []
            ori_armor_pred_per_img = []
            armor_num_per_img = []
            if pred[0].shape[0] == 0:
                img_preds.append([])
                car_locations.append([None, None])
                continue

            for i in range(pred[0].shape[0]):
                if (pred[0][i][5] == 0):
                    ori_car_pred_per_img.append(pred[0][i])

                    car_x1 = pred[0][i][0].cpu().numpy().astype(np.uint16) 
                    car_y1 = pred[0][i][1].cpu().numpy().astype(np.uint16)
                    car_x2 = pred[0][i][2].cpu().numpy().astype(np.uint16)
                    car_y2 = pred[0][i][3].cpu().numpy().astype(np.uint16)
                    car_acc = pred[0][i][4].cpu().numpy().astype(np.uint16)

                    # predict the number of armor
                    # crop as x1+1/4*width, x2-1/4*width for the armor_area to remove the LEDs for armor

                    ori_car_area = img0[car_y1:car_y2,
                                          car_x1:car_x2]
                    
                    car_area = ori_car_area.copy()
                    ori_shape_armor = (ori_car_area.shape[0], ori_car_area.shape[1], ori_car_area.shape[2])
                    cv2.imwrite('test_armor.png', ori_car_area*255)
                    cv2.imwrite('test_ori_armor.png', ori_car_area)

                    # input is row: 20 col: 28 format: CV_8UC1
                    ori_car_area_grey = (cv2.cvtColor(car_area, cv2.COLOR_BGR2GRAY) /
                         255.0).astype(np.float32)

                    car_area_img = letterbox(car_area, [640, 640], stride=32, auto=False)[
                        0]  # padded resize
                    car_area_img = car_area_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
                    car_area_img = np.ascontiguousarray(car_area_img)  # contiguous
                    car_area_img = torch.from_numpy(car_area_img).to(self.device).float()
                    car_area_img = car_area_img / 255
                    if len(car_area_img.shape) == 3:
                        car_area_img = car_area_img[None]  # expand for batch dim

                    # torch.Size([1, 3, 448, 640])
                    pred_armor = self.num_net(car_area_img)
                    # output of the pred: shape (b, N, 6)
                    # x1, y1, x2, y2, acc, class

                    pred_armor = non_max_suppression(pred_armor, conf_thres=0.25, iou_thres=0.45)
                    
                    pred_armor[0][:, :4] = scale_boxes(
                        car_area_img.shape[2:], pred_armor[0][:, :4], ori_shape_armor).round()
                    #transform the armor bbox to the original image
                    

                    

                    for k in range(pred_armor[0].shape[0]):
                        armor_num_per_img.append(pred_armor[0][k][5].cpu().numpy().astype(np.uint16))
                        ori_armor_pred_per_img.append(pred_armor[0][k])
                    

            car_index = 0

            for i in range(len(ori_car_pred_per_img)):
                car_x1 = ori_car_pred_per_img[i][0].cpu().numpy()
                car_y1 = ori_car_pred_per_img[i][1].cpu().numpy()
                car_x2 = ori_car_pred_per_img[i][2].cpu().numpy()
                car_y2 = ori_car_pred_per_img[i][3].cpu().numpy()
                car_acc = ori_car_pred_per_img[i][4].cpu().numpy()
                car_cls = int(ori_car_pred_per_img[i][5].cpu().numpy())
                car_num = -1

                for j in range(len(ori_armor_pred_per_img)):
                    armor_x1 = ori_armor_pred_per_img[j][0].cpu().numpy()+car_x1
                    armor_y1 = ori_armor_pred_per_img[j][1].cpu().numpy() + car_y1
                    armor_x2 = ori_armor_pred_per_img[j][2].cpu().numpy() + car_x1
                    armor_y2 = ori_armor_pred_per_img[j][3].cpu().numpy() + car_y1
                    armor_acc = ori_armor_pred_per_img[j][4].cpu().numpy()

                    # if the armor is in the range of the car then the car num is the armor num
                    if (armor_x1 >= car_x1) and (armor_y1 >= car_y1) and (armor_x2 <= car_x2) and (armor_y2 <= car_y2):
                        car_num = armor_num_per_img[j]
                        single_armor_pred = [armor_x1, armor_y1,
                                             armor_x1, armor_y2,
                                             armor_x2, armor_y1,
                                             armor_x2, armor_y2,
                                             armor_acc, car_num,
                                             car_index,
                                             armor_x1, armor_y1,
                                             armor_x2-armor_x1, armor_y2-armor_y1]

                        armor_pred_per_img.append(single_armor_pred)
                if car_num == -1:
                    continue
                car_index = car_index + 1
                single_car_pred = [car_num, float(car_acc),
                                   [float(car_x1), float(car_y1), float(car_x2), float(car_y2)]]
                car_bbox_pre_img.append([car_x1, car_y1, car_x2, car_y2])
                car_pred_per_img.append(single_car_pred)

            if len(car_pred_per_img) == 0:
                img_preds.append([])
                car_locations.append([None, None])
                continue

            img_preds.append(car_pred_per_img)
            car_locations.append([np.array(armor_pred_per_img).astype(np.float32),
                                 np.array(car_bbox_pre_img).astype(np.float32)])

            preds.append(pred)

        return img_preds, car_locations
