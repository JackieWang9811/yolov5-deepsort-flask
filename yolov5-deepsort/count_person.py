import numpy as np

import objtracker
from objdetector import Detector
import cv2
import time


VIDEO_PATH = './test.mp4'

if __name__ == '__main__':

    # 根据视频尺寸，填充供撞线计算使用的polygon(多边形)
    width = 1920
    height = 1080
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)

    # 填充第一个撞线polygon（蓝色）
    list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
                     [299, 375], [267, 289]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    # cv2.fillPoly 填充多边形
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    # np.newaxis增加一个维度，变成三维图像
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个撞线polygon（黄色）
    mask_image_temp = np.zeros((height, width), dtype=np.uint8)
    list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
                       [594, 637], [118, 483], [109, 303]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用的mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (width//2, height//2))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (width//2, height//2))

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 下行数量
    down_count = 0
    # 上行数量
    up_count = 0

    # cv2.FONT_HERSHEY_SIMPLEX
    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int((width/2) * 0.01), int((height/2) * 0.05))

    # 实例化yolov5检测器
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture(VIDEO_PATH)
    """
    capture = cv2.VideoCapture()# 如果括号里面是0，则打开摄像头，如果为路径，则打开路径视频
    ret,frame = capture.read() # 按帧读取视频，ret返回的是布尔值，代表是否正确读取帧，如果视频读到结尾，它就返回False。
                               # frame就是每一帧的图像，是一个三维矩阵
    """
    # 设置fps
    fps=0.0
    while True:
        # 检测图片的起始时间
        t1 = time.time()
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (width//2, height//2))

        list_bboxs = []
        # 更新跟踪器(deepsort)
        # objtracker.update返回的是image和bounding box
        output_image_frame, list_bboxs = objtracker.update(detector, im)
        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        # fps计算
        fps = (fps + (1. / (time.time() - t1))) / 2  # 此处的time.time()就是检测完这张图片的结束时间,除以2是为了和之前的fps求一个平均
        # print("fps= %.2f" % fps)
        # cv2.putText(图片,添加的文字,左上角坐标，字体，字体大小，颜色，字体粗细)
        frame = cv2.putText(output_image_frame, "fps= %.2f" % (fps), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox
                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))
                # 撞线的点
                y = y1_offset
                x = x1
                if polygon_mask_blue_and_yellow[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                    # 判断 黄polygon list里是否有此 track_id
                    # 有此track_id，则认为是 UP (上行)方向
                    if track_id in list_overlapping_yellow_polygon:
                        # 上行+1
                        up_count += 1
                        print(f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {up_count} | 上行id列表: {list_overlapping_yellow_polygon}')
                        # 删除 黄polygon list 中的此id
                        list_overlapping_yellow_polygon.remove(track_id)

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 DOWN（下行）方向
                    if track_id in list_overlapping_blue_polygon:
                        # 下行+1
                        down_count += 1
                        print(f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_count} | 下行id列表: {list_overlapping_blue_polygon}')
                        # 删除 蓝polygon list 中的此id
                        list_overlapping_blue_polygon.remove(track_id)
            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)

                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
            list_overlapping_all.clear()
            # 清空list
            list_bboxs.clear()
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()
            
        # 输出计数信息
        text_draw = 'DOWN: ' + str(down_count) + \
                    ' , UP: ' + str(up_count)
        # 文本绘制
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=0.75, color=(0, 0, 255), thickness=2)
        # cv2.imshow()作用是在GUI里显示一副图像，但是它的持续时间是由cv2.waitKey()决定的
        cv2.imshow('Counting Demo', output_image_frame)
        # 保留n帧的时间
        cv2.waitKey(100)

    capture.release()
    cv2.destroyAllWindows()
