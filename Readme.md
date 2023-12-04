# RM2023-Deep Radar

RM23 Deep-Learning- based Radar

RM23 Competion Logging Files [Download](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yhuangdl_connect_ust_hk/EfDgUZagcMBClsdbgzgLOo4BZ8fBIca_ndMtWQ7lpq5a5Q?e=Jt43m5)

### 一、功能介绍

Based on Open source project from SJTU RM21 Radar. Implemented with two cameras and Deep learning for depth prediction (ZoeDepth) and for armor detection (YoloV5).

### 二、效果展示

- UI界面

<img src="doc_imgs\missile.png" alt="missile" style="zoom:67%;" />

- 飞镖预警

<img src="doc_imgs\UI.jpg" alt="missile" style="zoom:67%;" />

- SJTU demo: b站[视频演示](https://www.bilibili.com/video/BV1FM4y1579H/)，以便大家熟悉我们雷达站测试的使用，附有讲解

### 三、环境配置

| 软件环境     | 硬件环境                  |
| ------------ | ------------------------- |
| Ubuntu 18.04 | NVIDIA GeForce GTX 1080Ti |
| ROS          | 鱼眼镜头                  |
| Miniconda    | HikVision camerasx2       |
| Pytorch/ONNX | USB转TTL                  |

1. [Download miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
2. Install python environment

```python
pip install -r requirements.txt
```

3. Adjust camera parameters:

   Hik Vision camera adjustment: [Parameters](https://blog.csdn.net/qq_42719217/article/details/125478983)

   Hik Vision Software: [机器视觉工业相机客户端MVS V2.1.2（Linux）](https://www.hikrobotics.com/cn/machinevision/service/download?module=0)
4. Download pretrained models:

   1. [Zoe-Depth](https://github.com/isl-org/ZoeDepth)
   2. [Number Recognition](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yhuangdl_connect_ust_hk/EYSl2wHpwMxJmDrgR5Wl8BgBtukBy5oc1CASQo34yczIWg?e=jbXNGG)
   3. [YoloV5 (trained on )](https://hkustconnect-my.sharepoint.com/:u:/g/personal/yhuangdl_connect_ust_hk/Ef9fgKP9EENAk0dCVFdb2KYBIvccnMH1J4QVBkSf60Lk5Q?e=8bZ3hv)
   4. Or train the yolo model by your self on the [dataset](https://github.com/SCAU-RM-NAV/RM2023_Radar_Dataset)

### 四、程序使用

#### demo运行方式

```
python two_cam_main.py
```

#### 神经网络特别说明

由于我们只提供了神经网络的预测结果，而没有提供神经网络本身（**attention！pkl不是神经网络，而是预测结果的录像**），所以需要用户自行添加神经网络

由于需要和我们的解析程序对应，需要用户自行编写神经网络预测结果与我们程序的adapter，即添加接口

```python
class Predictor(object):
    def __init__(self,weights = ""):
        '''
        :param weights:模型文件路径
        '''
        self.net = Network(weights)
    def transfer(results):
        raise NotImplementedError
        return img_preds,car_locations
    def infer(self,imgs):
        img_preds,car_locations = self.transfer(self.net.predict(img))
        return img_preds,car_locations
```

Adapter类的实例如上，这里假设用户的神经网络的预测结果是results，以上实例未添加实际的格式转换代码

```bash
[
# img_preds
[[['car_blue_2', 0.8359227180480957, [2330.0, 1371.0, 2590.0, 1617.0]]], []], 
# car_locations
[[array([[2.4680000e+03, 1.5520000e+03, 2.4690000e+03, 1.5720000e+03,
        2.5050000e+03, 1.5700000e+03, 2.5040000e+03, 1.5510000e+03,
        9.5630890e-01, 2.0000000e+00, 0.0000000e+00, 2.4680000e+03,
        1.5510000e+03, 3.7000000e+01, 2.1000000e+01],
       [2.3710000e+03, 1.5190000e+03, 2.3660000e+03, 1.5350000e+03,
        2.3850000e+03, 1.5450000e+03, 2.3890000e+03, 1.5280000e+03,
        9.2683524e-01, 2.0000000e+00, 0.0000000e+00, 2.3660000e+03,
        1.5190000e+03, 2.3000000e+01, 2.6000000e+01]], dtype=float32), array([[2330., 1371., 2590., 1617.]], dtype=float32)], [None, None]]
      
]
```

对于返回值 `img_preds`和 `car_locations`，以上是直接用python输出的结果，未做格式化，仅作为实例。

此外，我们提供以下的说明

##### img_preds

对于 `img_preds`是一个列表，每一项是对于一张图片的预测。每张图片的预测也是一个列表，每一项是一个目标的预测结果

```bash
[#图片级列表start
[#图片1start
['car_blue_2', 0.8359227180480957, [2330.0, 1371.0, 2590.0, 1617.0]] # 每项预测
]#图片1end
, 
#图片2start
[]
#图片2end
], #图片级列表end
```

| 名称   | 格式                                                                                     | 示例                             |
| ------ | ---------------------------------------------------------------------------------------- | -------------------------------- |
| 预测名 | str(若为车子则是"car\_{颜色}\_{编号（0为未识别编号）}"，若为其他则是"base"或者"watcher") | 'car_blue_2'                     |
| 置信度 | float                                                                                    | 0.8359227180480957               |
| bbox   | list[x0,y0（左上）,x1,y1（右下）]                                                        | [2330.0, 1371.0, 2590.0, 1617.0] |

**注意这里的预测是整个车，而不是装甲板**

##### car_locations

```bash
[#图片级列表start
    #图片1
    [
        # numpy数组（N*15） N为该图片中预测出的装甲板数
        array([
                [2.4680000e+03, 1.5520000e+03, 2.4690000e+03, 1.5720000e+03,
                2.5050000e+03, 1.5700000e+03, 2.5040000e+03, 1.5510000e+03,
                9.5630890e-01, 2.0000000e+00, 0.0000000e+00, 2.4680000e+03,
                1.5510000e+03, 3.7000000e+01, 2.1000000e+01],
                [2.3710000e+03, 1.5190000e+03, 2.3660000e+03, 1.5350000e+03,
                2.3850000e+03, 1.5450000e+03, 2.3890000e+03, 1.5280000e+03,
                9.2683524e-01, 2.0000000e+00, 0.0000000e+00, 2.3660000e+03,
                1.5190000e+03, 2.3000000e+01, 2.6000000e+01]
            ], dtype=float32), 
        # numpy数组（M*4） M为该图片中预测出的车辆（car，所以watcher和base不算）数，和上一个对应
        array([[2330., 1371., 2590., 1617.]], dtype=float32)
    ], 
  
    # 图片2 (没有任何装甲板预测为两个None)
    [None, None]

]# 图片级列表end
```

第一个numpy数组是装甲板预测结果，15维格式如下

| 名称                     | 维度    | 格式                                      | 示例                                                                                                                          |
| ------------------------ | ------- | ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 四点                     | （0:8） | （左上，左下，右下，右上）                | 2.4680000e+03, 1.5520000e+03, 2.4690000e+03, 1.5720000e+03,        2.5050000e+03, 1.5700000e+03, 2.5040000e+03, 1.5510000e+03 |
| 置信度                   | （8:9） | float                                     | 9.5630890e-01                                                                                                                 |
| 装甲板预测号             | (9:10)  | int(采用R1-R5编号为8-12,B1-B5为1-5来解析) | 2                                                                                                                             |
| 对应的车辆预测框号       | (10:11) | int                                       | 0                                                                                                                             |
| bbox(四点的最小外接矩阵) | (11:15) | （x0,y0,w,h)                              | 2.4680000e+03,1.5510000e+03, 3.7000000e+01, 2.1000000e+01                                                                     |

第二个numpy数组是该图片中预测出的车辆，4维格式如下

| 名称 | 维度  | 格式           | 示例                         |
| ---- | ----- | -------------- | ---------------------------- |
| bbox | (0:4) | （x0,y0,x1,y1) | [2330., 1371., 2590., 1617.] |

#### 相机配置文件

配置文件中camera后编号即对应程序中相机编号（camera_type) 0为右相机，1为左相机，2为上相机

##### Config文件

radar_class/config.py

##### yaml文件

yaml文件中保存相机标定参数,请标定后填入该文件（K_0为内参，C_0为畸变系数，E_0为雷达到相机外参）

### 五、文件目录

```
RM2023-DEEP-RADAR
│  .gitignore 
│  Demo_v4.py # PyQt5产生的UI设计python实现
│  LICENSE
│  mainEntry.py # 自定义UI类
│  main_l.py # 主程序
│  main_l.sh # 三个自启动脚本
│  start.sh
│  test.bash
│  Readme.md
│  requirements.txt # pip安装环境
│  UART.py # 裁判系统驱动
├─Camera # 相机参数
│      camera_0.Config
│      camera_1.Config
│      camera_2.Config
│    
├─Camerainfo # 相机标定参数
│      camera0.yaml
│      camera1.yaml
│      camera2.yaml
│    
├─demo_resource # 运行demo资源
│  │  demo_infer.pkl # 保存的神经网络预测文件
│  │  demo_pc.pkl # 保存的点云文件
│  │  demo_pic.jpg # 示例背景图
│  │  map2.jpg # 示例小地图
│  │  third_cam.mp4 # 示例上相机视频
│  │  
│  └─two_cam # 示例左右相机视频
│          1.mp4
│          2.mp4
│    
├─radar_class # 主要类
│     camera.py # 相机驱动类
│     common.py # 各类常用函数,包括绘制，装甲板去重
│     config.py # 配置文件
│     Lidar.py # 雷达驱动
│     location.py # 位姿估计
│     location_alarm.py # 位置预警类
│     missile_detect.py # 飞镖预警类
│     multiprocess_camera.py # 多进程相机类
│     network.py # 示例神经网络类
│     reproject.py # 反投影预警类
│     ui.py # hp及比赛阶段UI类
│        
├─serial_package # 官方的裁判系统驱动
│     Game_data_define.py
│     offical_Judge_Handler.py
│     init.py
│        
├─tools 
│      Demo_v4.ui # QtUI原始文件
│      generate_region.py # 产生感兴趣区域
│    
└─_sdk # mindvision相机驱动文件
       mvsdk.py
       init.py
```
