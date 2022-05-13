import numpy as np
import torch

# import .abc这个表示导入当前文件夹(abc)下的一个包（而不是导入其他文件夹的包）。
from .deep.feature_extractor import Extractor # 特征提取网络
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

# deepsort的整体封装，实现一个deepsort追踪的一个整体效果。
__all__ = ['DeepSort'] # __all__ 提供了暴露接口用的”白名单“


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence # 检测结果置信度阈值 
        self.nms_max_overlap = nms_max_overlap # 非极大抑制阈值，设置为1代表不进行抑制

        self.extractor = Extractor(model_path, use_cuda=use_cuda) # 用于提取一个batch图片对应的特征

        max_cosine_distance = max_dist  # 最大余弦距离，用于级联匹配，如果大于该阈值，则忽略
        nn_budget = 100  # 每个类别gallery最多的外观描述子的个数，如果超过，删除旧的
        # NearestNeighborDistanceMetric 最近邻距离度量
        # 对于每个目标，返回到目前为止已观察到的任何样本的最近距离（欧式或余弦）。
        # 由距离度量方法构造一个 Tracker。
        # 第一个参数可选'cosine' or 'euclidean'
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        # 图像是三维的，H*W*C，取出前两维
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        # 从原图中抠取bbox对应图片并计算得到相应的特征,ori_img代表原图
        features = self._get_features(bbox_xywh, ori_img)
        # (18,4代表18个行人，每个行人里面有四个信息)
        # 将bounding box的坐标格式进行转换，从xywh格式转换为tlwh格式
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)        
        # 筛选掉小于min_confidence的目标，并构造一个由Detection对象构成的列表
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        # 返回非极大值抑制以后的索引
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        # 将满足条件的对象放入列表中
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict() # 将跟踪状态分布向前传播一步
        #
        self.tracker.update(detections) # 执行测量更新和跟踪管理

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            #
            outputs = np.stack(outputs,axis=0)
        return outputs


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    #将bbox的[x,y,w,h] 转换成[t,l,w,h]
    @staticmethod # 静态方法可以无需实例化调用，不需要加self关键字，表明这个方法不需要用到对象中的任何资源，和普通的函数没什么区别
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh

    #将bbox的[x,y,w,h] 转换成[x1,y1,x2,y2]
    #某些数据集例如 pascal_voc 的标注方式是采用[x，y，w，h]
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    # 获取抠图部分的特征
    # 针对目标框的坐标位置，对原图像进行抠图
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2] # 抠图部分
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops) # 对抠图部分提取特征
        else:
            features = np.array([])
        return features


