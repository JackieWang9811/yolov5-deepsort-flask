from importlib import import_module
import os
from flask import Flask, render_template, Response
import numpy as np
import objtracker
from objdetector import Detector
import cv2
import time

VIDEO_PATH = './video/test_person.mp4'
# import camera driver
class VideoCamera(object):
    def __init__(self):
        # 根据视频尺寸，填充供撞线计算使用的polygon(多边形)
        self.width = 1920
        self.height = 1080
        self.mask_image_temp = np.zeros((self.height, self.width), dtype=np.uint8)

        # 填充第一个撞线polygon（蓝色）
        self.list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
                         [299, 375], [267, 289]]
        self.ndarray_pts_blue = np.array(self.list_pts_blue, np.int32)
        # cv2.fillPoly 填充多边形
        self.polygon_blue_value_1 = cv2.fillPoly(self.mask_image_temp, [self.ndarray_pts_blue], color=1)
        # np.newaxis增加一个维度，变成三维图像
        self.polygon_blue_value_1 = self.polygon_blue_value_1[:, :, np.newaxis]

        # 填充第二个撞线polygon（黄色）
        self.mask_image_temp = np.zeros((self.height, self.width), dtype=np.uint8)
        self.list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
                           [594, 637], [118, 483], [109, 303]]
        self.ndarray_pts_yellow = np.array(self.list_pts_yellow, np.int32)
        self.polygon_yellow_value_2 = cv2.fillPoly(self.mask_image_temp, [self.ndarray_pts_yellow], color=2)
        self.polygon_yellow_value_2 = self.polygon_yellow_value_2[:, :, np.newaxis]

        # 撞线检测用的mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
        self.polygon_mask_blue_and_yellow = self.polygon_blue_value_1 + self.polygon_yellow_value_2

        # 缩小尺寸，1920x1080->960x540
        self.polygon_mask_blue_and_yellow = cv2.resize(self.polygon_mask_blue_and_yellow, (self.width // 2, self.height // 2))

        # 蓝 色盘 b,g,r
        self.blue_color_plate = [255, 0, 0]
        # 蓝 polygon图片
        self.blue_image = np.array(self.polygon_blue_value_1 * self.blue_color_plate, np.uint8)

        # 黄 色盘
        self.yellow_color_plate = [0, 255, 255]
        # 黄 polygon图片
        self.yellow_image = np.array(self.polygon_yellow_value_2 * self.yellow_color_plate, np.uint8)

        # 彩色图片（值范围 0-255）
        self.color_polygons_image = self.blue_image + self.yellow_image

        # 缩小尺寸，1920x1080->960x540
        self.color_polygons_image = cv2.resize(self.color_polygons_image, (self.width // 2, self.height // 2))

        # list 与蓝色polygon重叠
        self.list_overlapping_blue_polygon = []

        # list 与黄色polygon重叠
        self.list_overlapping_yellow_polygon = []

        # 下行数量
        self.down_count = 0
        # 上行数量
        self.up_count = 0

        # cv2.FONT_HERSHEY_SIMPLEX
        self.font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
        self.draw_text_postion = (int((self.width / 2) * 0.01), int((self.height / 2) * 0.05))

        # 实例化yolov5检测器
        self.detector = Detector()

        # 打开视频
        self.capture = cv2.VideoCapture(VIDEO_PATH)
        """
        capture = cv2.VideoCapture()# 如果括号里面是0，则打开摄像头，如果为路径，则打开路径视频
        ret,frame = capture.read() # 按帧读取视频，ret返回的是布尔值，代表是否正确读取帧，如果视频读到结尾，它就返回False。
                                   # frame就是每一帧的图像，是一个三维矩阵
        """
        # 设置fps
        self.fps = 0.0
    def get_frame(self):
        while True:
            # 检测图片的起始时间
            t1 = time.time()
            # 读取每帧图片
            _, im = self.capture.read()
            if im is None:
                break

            # 缩小尺寸，1920x1080->960x540
            im = cv2.resize(im, (self.width // 2, self.height // 2))

            list_bboxs = []
            # 更新跟踪器(deepsort)
            # objtracker.update返回的是image和bounding box
            output_image_frame, list_bboxs = objtracker.update(self.detector, im)
            # 输出图片
            output_image_frame = cv2.add(output_image_frame, self.color_polygons_image)

            # fps计算
            self.fps = (self.fps + (1. / (time.time() - t1))) / 2  # 此处的 time.time()就是检测完这张图片的结束时间,除以2是为了和之前的fps求一个平均
            # print("fps= %.2f" % fps)
            # cv2.putText(图片,添加的文字,左上角坐标，字体，字体大小，颜色，字体粗细)
            frame = cv2.putText(output_image_frame, "fps= %.2f" % (self.fps), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                                2)

            if len(list_bboxs) > 0:
                # ----------------------判断撞线----------------------
                for item_bbox in list_bboxs:
                    x1, y1, x2, y2, label, track_id = item_bbox
                    # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                    y1_offset = int(y1 + ((y2 - y1) * 0.6))
                    # 撞线的点
                    y = y1_offset
                    x = x1
                    if self.polygon_mask_blue_and_yellow[y, x] == 1:
                        # 如果撞 蓝polygon
                        if track_id not in self.list_overlapping_blue_polygon:
                            self.list_overlapping_blue_polygon.append(track_id)
                        # 判断 黄polygon list里是否有此 track_id
                        # 有此track_id，则认为是 UP (上行)方向
                        if track_id in self.list_overlapping_yellow_polygon:
                            # 上行+1
                            self.up_count += 1
                            print(
                                f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {self.up_count} | 上行id列表: {self.list_overlapping_yellow_polygon}')
                            # 删除 黄polygon list 中的此id
                            self.list_overlapping_yellow_polygon.remove(track_id)

                    elif self.polygon_mask_blue_and_yellow[y, x] == 2:
                        # 如果撞 黄polygon
                        if track_id not in self.list_overlapping_yellow_polygon:
                            self.list_overlapping_yellow_polygon.append(track_id)
                        # 判断 蓝polygon list 里是否有此 track_id
                        # 有此 track_id，则 认为是 DOWN（下行）方向
                        if track_id in self.list_overlapping_blue_polygon:
                            # 下行+1
                            self.down_count += 1
                            print(
                                f'类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {self.down_count} | 下行id列表: {self.list_overlapping_blue_polygon}')
                            # 删除 蓝polygon list 中的此id
                            self.list_overlapping_blue_polygon.remove(track_id)
                # ----------------------清除无用id----------------------
                list_overlapping_all = self.list_overlapping_yellow_polygon + self.list_overlapping_blue_polygon
                for id1 in list_overlapping_all:
                    is_found = False
                    for _, _, _, _, _, bbox_id in list_bboxs:
                        if bbox_id == id1:
                            is_found = True
                    if not is_found:
                        # 如果没找到，删除id
                        if id1 in self.list_overlapping_yellow_polygon:
                            self.list_overlapping_yellow_polygon.remove(id1)

                        if id1 in self.list_overlapping_blue_polygon:
                            self.list_overlapping_blue_polygon.remove(id1)
                list_overlapping_all.clear()
                # 清空list
                list_bboxs.clear()
            else:
                # 如果图像中没有任何的bbox，则清空list
                self.list_overlapping_blue_polygon.clear()
                self.list_overlapping_yellow_polygon.clear()

            # 输出计数信息
            text_draw = 'DOWN: ' + str(self.down_count) + \
                        ' , UP: ' + str(self.up_count)
            # 文本绘制
            output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                             org=self.draw_text_postion,
                                             fontFace=self.font_draw_number,
                                             fontScale=0.75, color=(0, 0, 255), thickness=2)
            ret, jpeg = cv2.imencode('.jpg', output_image_frame)
            return jpeg.tobytes()

    # def get_frame(self):

        # for i in range(50):
        #     success, image = self.video.read()
        # image= detect(source=image,half=self.half,model=self.model,device=self.device,imgsz=self.imgsz,stride=self.stride)

        # ret,jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()

app = Flask(__name__)


@app.route('/') # 定义起始页
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        # frame = camera.get_frame()
        # 这里应该循环赋值检测结果
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')  # 定义路由
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    # Response向前端返回数据
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # app.run 提供一个web服务，host代表监听的网段
    app.run(host='0.0.0.0', threaded=True, port=5001)
