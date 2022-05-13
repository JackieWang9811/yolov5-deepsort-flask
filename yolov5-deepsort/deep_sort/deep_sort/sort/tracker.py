# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
"""
保存了所有的轨迹信息，负责初始化第一帧，卡尔曼滤波的预测和更新，负责级联匹配，IOU匹配
"""

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
        测量与轨迹关联的距离度量
    max_age : int
        Maximum number of missed misses before a track is deleted.
        删除轨迹前的最大未命中数
    n_init : int
        Number of frames that a track remains in initialization phase.
        确认轨迹前的连续检测次数。如果前n_init帧内发生未命中，则将轨迹状态设置为Deleted
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter() # 实例化卡尔曼滤波器
        self.tracks = []   # 保存一个轨迹列表，用于保存一系列轨迹
        self._next_id = 1  # 下一个分配的轨迹id
 
    def predict(self):
        """Propagate track state distributions one time step forward.
        将跟踪状态分布向前传播一步

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.
        执行测量更新和轨迹管理

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        # 进行级联匹配+IOU匹配
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        
        # 1. 针对匹配上的结果（matches）
        for track_idx, detection_idx in matches:
            # 更新tracks中相应的detection
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        
        # 2. 针对未匹配的track（unmatched_tracks）, 调用mark_missed进行标记
        # track失配时，若待定则删除；若update时间很久也删除
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # 3. 针对未匹配的detection（unmatched_detections）,detection失配，进行初始化
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        # 得到最新的tracks列表，保存的是标记为Confirmed和Tentative的track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        # 使用新的track，track ID 以及track features来更新特征矩阵，便于下一帧计算
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            # 获取所有Confirmed状态的track id
            if not track.is_confirmed():
                continue
            features += track.features  # 将Confirmed状态的track的features添加到features列表
            # 获取每个feature对应的track id
            targets += [track.track_id for _ in track.features]
            track.features = []
        # 距离度量中的特征集更新
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    # 主要功能是进行匹配，找到匹配的，未匹配的部分
    def _match(self, detections):

        # 用于计算tarck和detections之间的距离
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            
            # 通过最近邻（余弦距离）计算出代价矩阵 cost matrix
            # 这里没有用到马氏距离
            cost_matrix = self.metric.distance(features, targets)
            # 计算门控后的成本矩阵（代价矩阵）
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # 区分开confirmed tracks和unconfirmed tracks
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        """
        Matching Cascade
        级联匹配（仅针对确认态）
        """
        # 对确定态的轨迹进行级联匹配（Matching Cascade）
        # matching_cascade 根据特征将检测框匹配到确认的轨迹。
        # 传入门控后的成本矩阵
        # 进行级联匹配后，会产生三种状态，
        # 一个是unmatched_tracks_a：未匹配的跟踪框，即目标从检测框中小时，但跟踪框（预测框）中仍然存在该目标
        # 一个是unmatched_detections：未匹配的检测框：检测框中新增的目标，但跟踪框（预测框）中不存在该新增目标
        # 一个是matches_a：匹配的跟踪框（预测框）和检测框
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        """
        IOU Assignment
        IOU匹配
        """
        # 将未确定态的轨迹和刚刚没有匹配上的轨迹组合为 iou_track_candidates 
        # 并进行基于IoU的匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1] # 刚刚没有匹配上的轨迹
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1] # 已经很久没有匹配上的轨迹
        # 对级联匹配中还没有匹配成功的目标再进行IoU匹配
        # min_cost_matching 使用匈牙利算法解决线性分配问题。
        # 传入 iou_cost，尝试关联剩余的轨迹与未确认的轨迹。
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b # 组合两部分匹配 
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        # 得到再次处理得到的matches，unmatched_tracks，unmatched_detections
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
