# -- coding: utf-8 --

import os
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *


def hik_init(connection_num=0):
    SDKVersion = MvCamera.MV_CC_GetSDKVersion()
    print("SDKVersion[0x%x]" % SDKVersion)

    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        sys.exit()

    if deviceList.nDeviceNum == 0:
        print("find no device!")
        sys.exit()

    print("Find %d devices!" % deviceList.nDeviceNum)

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(
            MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            nip1 = (
                (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = (
                (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = (
                (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)

    
    # 0 is the left cam, 1 is the right cam

    nConnectionNum = connection_num

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄| en:Select device and create handle
    stDeviceList = cast(deviceList.pDeviceInfo[int(
        nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)

    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()



    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    ret = cam.MV_CC_SetFloatValue("ExposureTime", 200000.0000)
    if ret != 0:
        print("Set ExposureTime fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:开始取流 | en:Start grab image
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:获取数据包大小 | en:Get payload size
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))

    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()

    nPayloadSize = stParam.nCurValue
    data_buf = (c_ubyte * nPayloadSize)()  # image buffer
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

    info_lst = [cam, data_buf, nPayloadSize, stFrameInfo]
    # check frame readability
    for i in range(5):
        ret = cam.MV_CC_GetOneFrameTimeout(
            data_buf, nPayloadSize, stFrameInfo, 1000)
        if ret != 0:
            print("pipline broke while testing frame readability" % ret)
            sys.exit()

    return info_lst
    # -------------------------------------------

def letterbox_image(image, size):
    # 对图片进行resize，使图片不失真。在空缺的地方进行padding
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def image_show(images, name, num, w=3072, h=2048):
    img = Image.fromarray(images.astype('uint8')).convert('RGB')
    image = letterbox_image(img, (w, h))
    image = np.array(image)
    save_dir_path = 'result'
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)
    file_name = save_dir_path + '/' + str(num) + '.jpg'
    cv2.imwrite(file_name, image)

def image_control(data, stFrameInfo, receivedFrameCount):
    if stFrameInfo.enPixelType == 17301505:
        image = data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
        # image_show(images=image, name=stFrameInfo.nHeight, num=receivedFrameCount)
    elif stFrameInfo.enPixelType == 17301514:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_GB2RGB)
        # image_show(images=image, name=stFrameInfo.nHeight, num=stFrameInfo.nFrameNum)
    elif stFrameInfo.enPixelType == 35127316:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        # image_show(images=image, name=stFrameInfo.nHeight, num=stFrameInfo.nFrameNum)
    elif stFrameInfo.enPixelType == 34603039:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_Y422)
        # image_show(images=image, name=stFrameInfo.nHeight, num=stFrameInfo.nFrameNum)
    elif stFrameInfo.enPixelType == 34603058:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_YUV2BGR_YUYV)
        # image_show(images=image, name=stFrameInfo.nHeight, num=stFrameInfo.nFrameNum)
    elif stFrameInfo.enPixelType == 17301513:
        data = data.reshape(stFrameInfo.nHeight, stFrameInfo.nWidth, -1)
        image = cv2.cvtColor(data, cv2.COLOR_BAYER_RG2RGB)

    return image


def read_hik_frame(info_lst):
    cam = info_lst[0]
    data_buf = info_lst[1]
    nPayloadSize = info_lst[2]
    stFrameInfo = info_lst[3]

    ret = cam.MV_CC_GetOneFrameTimeout(
        data_buf, nPayloadSize, stFrameInfo, 1000)
    if ret == 0:
        # print("get one frame: Width[%d], Height[%d], nFrameNum[%d], enPixelType[%d]" % (
            # stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum, stFrameInfo.enPixelType))
        receivedFrameCount = 0
        image = np.asarray(data_buf).reshape(
            (stFrameInfo.nHeight, stFrameInfo.nWidth, -1))
        image = image_control(data=image, stFrameInfo=stFrameInfo, receivedFrameCount=receivedFrameCount)
        image = cv2.resize(image, (1080, 720))

        # use this to show image
        # cv2.imshow("show", image)
        k = cv2.waitKey(1) & 0xff
        return True, image
    else:
        print("no data[0x%x] --- Hik" % ret)
        return False, None


def hik_close(info_lst):
    cam = info_lst[0]
    data_buf = info_lst[1]
    # ch:停止取流 | en:Stop grab image
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:关闭设备 | Close device
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close deivce fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()

    # ch:销毁句柄 | Destroy handle
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)
        del data_buf
        sys.exit()
    del data_buf

if __name__ == "__main__":
    info_lst = hik_init()
    while True:
        read_hik_frame(info_lst)

    hik_close(info_lst)