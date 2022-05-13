"""
封装的yolov5检测器脚本
"""
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.torch_utils import select_device
import cv2

import objtracker

# 使用的权重是在coco数据集上预训练的，可以检测80个物体
DETECTOR_PATH = 'weights/best.pt'
# 可以选择其中一些需要预测的目标进行检测
OBJ_LIST = ['person', 'car', 'bus', 'truck']


class baseDet(object):
    def __init__(self):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1

    def build_config(self):
        self.frameCounter = 0

    def feedCap(self, im, func_status):
        # 初始化dict，用于返回结果，在这里初始化了键list_of_ids，但是后面没用到
        retDict = {
            'frame': None,
            'list_of_ids': None,
            'obj_bboxes': []
        }
        # 帧数统计
        self.frameCounter += 1

        im, obj_bboxes = objtracker.update(self, im)
        cv2.imshow('test', im)
        retDict['frame'] = im
        retDict['obj_bboxes'] = obj_bboxes

        return retDict

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")

# 这个类就是对yolov5检测器的一个封装
class Detector(baseDet):
    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):
        self.weights = DETECTOR_PATH
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        # print(self.device)
        self.device = select_device(self.device)
        # 加载权重并构造实例
        model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        model.to(self.device).eval()
        # 将模型参数从float32类型转变为float16类型
        model.half()
        self.m = model
        # hasattr() 函数用于判断对象是否包含对应的属性。
        self.names = model.module.names if hasattr(   # get class names
            model, 'module') else model.names
        # print(self.names)

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # img0是原始图像，img是经过预处理后的图像
        return img0, img

    def detect(self, im):
        im0, img = self.preprocess(im)
        # 将图片传入模型，得到预测结果
        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)
        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                # 将 det得到的坐标从img重新错放为im0的形状，并做四舍五入
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    #
                    if not lbl in OBJ_LIST:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
        return im, pred_boxes

