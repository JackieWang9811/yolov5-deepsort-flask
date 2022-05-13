  REID_CKPT: "deep_sort/deep_sort/deep/checkpoint/ckpt.t7" # deepsort 特征提取网络权重的目录路径
  MAX_DIST: 0.2 # 最大余弦距离，用于级联匹配，如果大于该阈值，则忽略
  MIN_CONFIDENCE: 0.3 # 检测结果置信度阈值，
  NMS_MAX_OVERLAP: 0.5 # 非极大值抑制
  MAX_IOU_DISTANCE: 0.7 # 最大IOU阈值
  MAX_AGE: 70 # 最大寿命，也就是经过MAX_AGE帧没有追踪到该物体，就将该轨迹变为删除态
  N_INIT: 3 # 最高击中次数，如果击中该次数，就由不确定态转换为确定态
  NN_BUDGET: 100 # 最大保存特征帧数，如果超过该帧数，将进行滚动保存