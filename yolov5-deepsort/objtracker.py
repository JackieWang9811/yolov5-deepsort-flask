"""
封装的deepsort跟踪器脚本
"""
import time

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import numpy as np

cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
# 对象实例化
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    list_pts = []
    point_radius = 4
    
    for (x1, y1, x2, y2, cls_id, pos_id) in bboxes:
        if cls_id in ['smoke', 'phone', 'eat']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if cls_id == 'eat':
            cls_id = 'eat-drink'
            
        # check whether hit line 
        check_point_x = x1
        check_point_y = int(y1 + ((y2 - y1) * 0.6))

        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 5,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                    
        list_pts.append([check_point_x-point_radius, check_point_y-point_radius])
        list_pts.append([check_point_x-point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y+point_radius])
        list_pts.append([check_point_x+point_radius, check_point_y-point_radius])

        ndarray_pts = np.array(list_pts, np.int32)
        cv2.fillPoly(image, [ndarray_pts], color=(0, 0, 255))
        list_pts.clear()
    return image

def update(target_detector, image):
        start_time = time.time()
        # 返回的boxes里面包含了检测到的目标的个数，以及每一个目标的目标框左上角以及右下角坐标，class类别以及置信度
        _, bboxes = target_detector.detect(image)
        end_time = time.time()
        print("yolov5检测时间为:{}ms".format((end_time-start_time)*1000))
        bbox_xywh = []
        confs = []
        bboxes2draw = []
        if len(bboxes):
            # Adapt detections to deep sort input format 将所有检测结果转换成deepsort格式
            for x1, y1, x2, y2, label, conf in bboxes:
                # deepsort输入的是中心点坐标以及宽高
                obj = [
                    int((x1+x2)/2), int((y1+y2)/2),
                    x2-x1, y2-y1
                ]
                bbox_xywh.append(obj)
                confs.append(conf)
                # torch.Tensor()传入的是一个列表
            xywhs = torch.Tensor(bbox_xywh) # bbox_xywh检测框的中心点坐标以及宽高
            confss = torch.Tensor(confs) # 置信度

            # Pass detections to deepsort：将检测结果传递给deepsort
            outputs = deepsort.update(xywhs, confss, image)
            for value in list(outputs):
                # 左上角坐标，右下角坐标以及跟踪id
                x1,y1,x2,y2,track_id = value
                # 计算中心点
                center_x = (x1 + x2) * 0.5
                center_y = (y1 + y2) * 0.5

                # label = search_label(center_x=center_x, center_y=center_y,
                #                      bboxes_xyxy=bboxes, max_dist_threshold=20.0)
                bboxes2draw.append(
                    (x1, y1, x2, y2, label, track_id)
                )
        image = plot_bboxes(image, bboxes2draw)
        return image, bboxes2draw

def search_label(center_x, center_y, bboxes_xyxy, max_dist_threshold):
    """
    在 yolov5 的 bbox 中搜索中心点最接近的label
    :param center_x:
    :param center_y:
    :param bboxes_xyxy:
    :param max_dist_threshold:
    :return: 字符串
    """
    label = ''
    # min_label = ''
    min_dist = -1.0

    for x1, y1, x2, y2, lbl, conf in bboxes_xyxy:
        center_x2 = (x1 + x2) * 0.5
        center_y2 = (y1 + y2) * 0.5

        # 横纵距离都小于 max_dist
        min_x = abs(center_x2 - center_x)
        min_y = abs(center_y2 - center_y)

        if min_x < max_dist_threshold and min_y < max_dist_threshold:
            # 距离阈值，判断是否在允许误差范围内
            # 取 x, y 方向上的距离平均值
            avg_dist = (min_x + min_y) * 0.5
            if min_dist == -1.0:
                # 第一次赋值
                min_dist = avg_dist
                # 赋值label
                label = lbl
                pass
            else:
                # 若不是第一次，则距离小的优先
                if avg_dist < min_dist:
                    min_dist = avg_dist
                    # label
                    label = lbl
                pass
            pass
        pass

    return label
