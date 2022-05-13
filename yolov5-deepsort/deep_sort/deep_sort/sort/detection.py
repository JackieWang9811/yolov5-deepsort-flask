# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.
    此类表示单个图像中的边界框检测
    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : float
    # 置信度
        Detector confidence score.
    feature : array_like
    # 描述此图像中包含的对象的特征向量。
        A feature vector that describes the object contained in this image.
    Attributes
    ----------
    tlwh : ndarray
        t: top left x l: top left y w:width h:height
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """
    def __init__(self, tlwh, confidence, feature):
        # 实例化
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    # 这个没用到
    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        # 将bouding box的格式转换为(中心点x，中心点y，宽高比、以及高度)
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
