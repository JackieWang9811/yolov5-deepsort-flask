from objdetector import Detector
import imutils
import cv2

VIDEO_PATH = './video/test_person.mp4'
RESULT_PATH = 'result.mp4'

def main():

    func_status = {}
    func_status['headpose'] = None
    
    name = 'demo'

    det = Detector()
    cap = cv2.VideoCapture(VIDEO_PATH)
    # VideoCapture.get(5) 获取视频帧数
    fps = int(cap.get(5)) #30
    print('fps:', fps)
    t = int(1000/fps) #33

    size = None
    videoWriter = None

    while True:
        # 直到第三帧的时候才会显示检测结果和追踪结果
        # try:
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im, func_status)
        result = result['frame']
        result = imutils.resize(result, height=500)
        if videoWriter is None:
            # cv2.VideoWriter_fourcc是视频编解码器， fourcc意为四个字符代码（Four-Character Codes )，该编码由四个字符组成，
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            # VideoWriter(filename, fourcc, fps, frameSize, isColor), filename:保存的文件的路径, fourcc:指定编码器,
            # fps要保存的视频的帧率, frameSize 要保存的文件的画面尺寸, isColor 指示是黑白画面还是彩色的画面
            videoWriter = cv2.VideoWriter(
                RESULT_PATH, fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break

    # video结束时的处理
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()