import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Zoe_N
class DepthPredictor(object):
    def __init__(self,K_0,C_0,size):
        '''
        :param size: image size [W,H]
        :param K_0: 相机内参
        :param C_0: 畸变系数
        '''
        self.size = size
        self.K_0 = K_0
        self.C_0 = C_0
        model_zoe_n = torch.hub.load("./ZoeDepth", "ZoeD_N", source='local', pretrained=True)

        ##### sample prediction
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.zoe = model_zoe_n.to(DEVICE)

    def infer(self, img):
        # Local file
        # RGB format
        depth_numpy = self.zoe.infer_pil(img)  # as numpy
        # print(depth_numpy)
        # print(depth_numpy.min(), depth_numpy.max())
        # print(depth_numpy.shape)
        
        
        # figure = plt.figure()
        # plt.imshow(depth_numpy)
        # plt.axis('off')
        # plt.margins(0,0)
        # figure.set_tight_layout(True)
        # plt.tight_layout()
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # figure.savefig('test.png', bbox_inches='tight', pad_inches = 0)

        # figure1 = plt.figure()
        # plt.imshow(img)
        # plt.axis('off')
        # plt.margins(0,0)
        # figure1.set_tight_layout(True)
        # plt.tight_layout()
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # figure1.savefig('ori.png', bbox_inches='tight', pad_inches = 0)
        
        # plt.cla()
        # plt.close("all")
        # exit()
        return depth_numpy

    def depth_detect_refine(self,r,depth):
        '''
        :param r: the bounding box of armor , format (x0,y0,w,h)

        :return: (x0,y0,z) x0,y0是中心点在归一化相机平面的坐标前两位，z为其对应在相机坐标系中的z坐标值
        : the x0, y0 is the normalized points where x0 = X0 * width, y0 = Y0 * height
        '''
        center = np.float32([r[0]+r[2]/2,r[1]+r[3]/2])
        # 采用以中心点为基准点扩大一倍的装甲板框，并设置ROI上界和下界，防止其超出像素平面范围
        area = depth[int(max(0,center[1]-r[3])):int(min(center[1]+r[3],self.size[1]-1)),
               int(max(center[0]-r[2],0)):int(min(center[0]+r[2],self.size[0]-1))]
        z = np.nanmean(area) if not np.isnan(area).all() else np.nan # 当对应ROI全为nan，则直接返回为nan
        return np.concatenate([cv2.undistortPoints(center, self.K_0, self.C_0).reshape(-1),np.array([z])],axis = 0)
    
    def detect_depth(self,rects,depth):
        '''
        :param rects: List of the armor bounding box with format (x0,y0,w,h)

        :return: an array, the first dimension is the amount of armors input, and the second is the location data (x0,y0,z)
        x0,y0是中心点在归一化相机平面的坐标前两位，z为其对应在相机坐标系中的z坐标值
        '''
        if len(rects) == 0:
            return []

        ops = []
        for rect in rects:
             ops.append(self.depth_detect_refine(rect,depth))

        return np.stack(ops,axis = 0)
    
if __name__ == '__main__':
    dp = DepthPredictor(None, None, None)