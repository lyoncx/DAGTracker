import cv2
from scipy.spatial import KDTree
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import cosine_similarity
import pulp
from collections import defaultdict
import os
import torch.nn.functional as F
import torch
import re
import matplotlib.pyplot as plt

class TrackSubsegmentTuner:
    """
    增强版阈值分析器：修复了 GT 格式读取错误，并支持分析噪声导致的低分正确点。
    """
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.sample_points = []
        self.track_stats = []

    def calculate_iou(self, box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
        y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        union_area = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def collect_data(self, video_id, res_data, video_gt_dict):
        """
        res_data: List of [frame, id, x, y, w, h, conf, ...]
        video_gt_dict: {frame_id: [[gt_id, x, y, w, h], ...]} 或 {frame_id: {gt_id: [x, y, w, h]}}
        """
        tracks = defaultdict(list)
        for line in res_data:
            tracks[int(line[1])].append(line)

        for track_id, nodes in tracks.items():
            nodes.sort(key=lambda x: x[0])
            
            track_match_sequence = []
            for node in nodes:
                frame, _, x, y, w, h, conf = node[:7]
                curr_box = [x, y, w, h]
                
                best_iou = 0
                frame_gts = video_gt_dict.get(int(frame), [])
                
                # 核心修复逻辑：判断 frame_gts 的类型
                if isinstance(frame_gts, dict):
                    # 格式为 {gt_id: [x, y, w, h]}
                    for gt_id, gt_box in frame_gts.items():
                        iou = self.calculate_iou(curr_box, gt_box)
                        if iou > best_iou: best_iou = iou
                elif isinstance(frame_gts, list):
                    # 格式为 [[gt_id, x, y, w, h], ...] 或 [[x, y, w, h], ...]
                    for gt_item in frame_gts:
                        # 自动判断列表内元素格式
                        gt_box = gt_item[1:5] if len(gt_item) >= 5 else gt_item[:4]
                        iou = self.calculate_iou(curr_box, gt_box)
                        if iou > best_iou: best_iou = iou
                
                is_tp = 1 if best_iou >= self.iou_threshold else 0
                track_match_sequence.append({'conf': conf, 'is_tp': is_tp})
                self.sample_points.append((conf, is_tp))

            self.track_stats.append(track_match_sequence)

    def analyze(self):
        if not self.sample_points:
            print("没有收集到有效样本数据！请检查 GT 是否匹配。")
            return 0.4

        thresholds = np.arange(0.1, 0.95, 0.05)
        print(f"\n{'Threshold':<12} | {'Total Points':<12} | {'Precision':<12} | {'Status'}")
        print("-" * 65)

        recommended_seed = 0.4
        found_seed = False

        for thresh in thresholds:
            filtered_points = [p for p in self.sample_points if p[0] >= thresh]
            if not filtered_points: continue
            
            tps = sum(p[1] for p in filtered_points)
            precision = tps / len(filtered_points)
            
            # 如果准确率 > 98%，说明此阈值非常安全，可以作为轨迹起始的种子
            status = "SAFE SEED" if precision > 0.98 else "NOISY"
            
            print(f"{thresh:.2f}         | {len(filtered_points):<12} | {precision:.4f}     | {status}")
            
            if precision > 0.98 and not found_seed:
                recommended_seed = thresh
                found_seed = True

        print("-" * 65)
        print(f"推荐 Seed 阈值: {recommended_seed:.2f}")
        return recommended_seed

class GlobalTrajectoryAnalyzer:
    def __init__(self, iou_threshold=0.3, min_hit_rate=0.5):
        self.iou_threshold = iou_threshold
        self.min_hit_rate = min_hit_rate
        # 存储格式: [(score, is_tp), ...]
        self.all_track_results = []

    def calculate_iou(self, box_a, box_b):
        """
        box_a: [x1, y1, w, h] (tlwh)
        box_b: [x1, y1, x2, y2] (xyxy)
        """
        # Convert a to xyxy
        a_x1, a_y1, a_x2, a_y2 = box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]
        
        xA = max(a_x1, box_b[0])
        yA = max(a_y1, box_b[1])
        xB = min(a_x2, box_b[2])
        yB = min(a_y2, box_b[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (a_x2 - a_x1) * (a_y2 - a_y1)
        boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def add_video_data(self, video_id, baseline_data, video_gt_dict):
        """
        将单个视频的轨迹判定结果存入全局缓存
        baseline_data: list of [frame, id, x1, y1, w, h, score, ...]
        """
        if not baseline_data: return
        
        gt_data = video_gt_dict.get(video_id, {})
        if not gt_data: return

        # 1. 组织轨迹
        tracks = defaultdict(list)
        for item in baseline_data:
            tid = item[1]
            tracks[tid].append(item)

        # 2. 判定每条轨迹是否为 TP
        for tid, points in tracks.items():
            match_hits = 0
            scores = []
            
            for p in points:
                fid = int(p[0])
                pred_tlwh = [p[2], p[3], p[4], p[5]]
                scores.append(p[6]) # 提取该帧得分
                
                frame_gts = gt_data.get(fid, [])
                best_iou = 0
                for gt in frame_gts:
                    # gt: [id, x, y, w, h]
                    gt_xyxy = [gt[1], gt[2], gt[1]+gt[3], gt[2]+gt[4]]
                    iou = self.calculate_iou(pred_tlwh, gt_xyxy)
                    if iou > best_iou: best_iou = iou
                
                if best_iou > self.iou_threshold:
                    match_hits += 1
            
            # 轨迹层面的 TP 判定条件
            is_tp = (match_hits / len(points)) >= self.min_hit_rate
            # 轨迹的代表得分（取平均值）
            avg_score = np.mean(scores)
            
            self.all_track_results.append((avg_score, is_tp))

    def run_full_analysis(self):
        """
        计算最佳阈值并输出统计报告
        """
        if not self.all_track_results:
            print("没有可供分析的轨迹数据！")
            return

        # 按得分从高到低排序
        self.all_track_results.sort(key=lambda x: x[0], reverse=True)
        
        scores = np.array([x[0] for x in self.all_track_results])
        tp_flags = np.array([x[1] for x in self.all_track_results])
        
        total_positives = np.sum(tp_flags)
        total_tracks = len(tp_flags)
        
        print(f"\n汇总分析完成: 共检测到 {total_tracks} 条有效轨迹")
        print(f"其中 TP 轨迹 (基于当前匹配规则): {total_positives}")
        print("-" * 50)
        print(f"{'Threshold':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        
        best_f1 = -1
        best_thresh = 0
        
        # 遍历可能的阈值
        thresholds = np.linspace(0.05, 0.95, 19)
        for thresh in thresholds:
            # 在当前阈值下，哪些轨迹被保留
            keep_mask = scores >= thresh
            tp_count = np.sum(tp_flags[keep_mask])
            fp_count = np.sum(~tp_flags[keep_mask])
            
            precision = tp_count / (tp_count + fp_count + 1e-6)
            recall = tp_count / (total_positives + 1e-6) # 这里 Recall 是相对于所有能匹配上的 TP
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
            print(f"{thresh:<12.2f} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")

        print("-" * 50)
        print(f"推荐最佳置信度阈值: {best_thresh:.2f}")
        print(f"最高 F1-Score: {best_f1:.4f}")
        print("=" * 50)
        
        return best_thresh

class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        """
        初始化卡尔曼滤波器，支持动态状态维度。
        """
        # 状态向量 (dim_x 维)
        self.x = np.zeros(dim_x, dtype=np.float32)
        
        # 协方差矩阵 (dim_x * dim_x)
        self.P = np.eye(dim_x, dtype=np.float32)
        
        # 状态转移矩阵 (dim_x * dim_x)
        self.F = np.eye(dim_x, dtype=np.float32) 
        
        # 观测矩阵 (dim_z * dim_x)
        self.H = np.zeros((dim_z, dim_x), dtype=np.float32)
        
        # 过程噪声协方差 (dim_x * dim_x)
        self.Q = np.eye(dim_x, dtype=np.float32)
        
        # 观测噪声协方差 (dim_z * dim_z)
        self.R = np.eye(dim_z, dtype=np.float32)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.H @ self.x 

    def update(self, measurement):
        # 【关键修复】自动处理 (dim_z, 1) 这种列向量输入，将其转换为 (dim_z,)
        if measurement.ndim == 2 and measurement.shape[1] == 1:
            measurement = np.squeeze(measurement)
            
        # measurement: [dim_z] (1D array)
        # self.H @ self.x: [dim_z] (1D array)
        # 此时减法是逐元素的，y 保持为 1D array
        y = measurement - (self.H @ self.x)
        
        S = self.H @ self.P @ self.H.T + self.R
        
        # 确保 S 是可逆的
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: S matrix is singular. Kalman gain cannot be computed.")
            return

        K = self.P @ self.H.T @ S_inv
        
        # K: (dim_x, dim_z), y: (dim_z,) -> K @ y: (dim_x,)
        self.x += K @ y
        
        I = np.eye(self.x.shape[0], dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

def forward_filter(measurements, kf):
    """
    measurements: list of (x,y)
    返回:
      fwd_states: [N,4]
      fwd_covs  : [N,4,4]
    """
    fwd_states = []
    fwd_covs   = []
    x = kf.state.copy()
    P = kf.P.copy()
    
    for z in measurements:
        # predict
        x = kf.F @ x
        P = kf.F @ P @ kf.F.T + kf.Q
        # update
        y = z - (kf.H @ x)
        S = kf.H @ P @ kf.H.T + kf.R
        K = P @ kf.H.T @ np.linalg.inv(S)
        x = x + K @ y
        I = np.eye(4, dtype=np.float32)
        P = (I - K @ kf.H) @ P
        
        fwd_states.append(x.copy())
        fwd_covs.append(P.copy())
    return fwd_states, fwd_covs

def rts_smoother(fwd_states, fwd_covs, kf):
    """
    简易 RTS 回溯平滑
    fwd_states, fwd_covs: 前向滤波结果
    kf: KalmanFilter实例 (主要用F,Q)
    """
    N = len(fwd_states)
    smooth_states = [None]*N
    smooth_covs   = [None]*N
    
    smooth_states[-1] = fwd_states[-1].copy()
    smooth_covs[-1]   = fwd_covs[-1].copy()
    
    for k in range(N-2, -1, -1):
        xk  = fwd_states[k]
        Pk  = fwd_covs[k]
        xk1 = fwd_states[k+1]
        Pk1 = fwd_covs[k+1]
        
        A = Pk @ kf.F.T @ np.linalg.inv(kf.F @ Pk @ kf.F.T + kf.Q)
        smooth_states[k] = xk + A @ (smooth_states[k+1] - xk1)
        smooth_covs[k]   = Pk + A @ (smooth_covs[k+1] - Pk1) @ A.T
    
    return smooth_states, smooth_covs

class Track:
    """
    插入虚拟节点时使用
    """
    def __init__(self, track_id, init_pos, init_vel):
        self.track_id = track_id
        self.kf = KalmanFilter()
        self.kf.state[:2] = init_pos
        self.kf.state[2:] = init_vel

    def update(self, pos):
        self.kf.update(pos)

    def predict(self):
        return self.kf.predict()  # [x,y]

    @property
    def velocity(self):
        return self.kf.state[2:4]

def offline_smooth_track(measurements):
    """
    measurements: list of (x,y), len=N
    返回 smooth_states [N,4]
    """
    # 1) 构造临时 KF
    temp_kf = KalmanFilter()
    # 2) forward
    fwd_states, fwd_covs = forward_filter(measurements, temp_kf)
    # 3) RTS
    smooth_states, smooth_covs = rts_smoother(fwd_states, fwd_covs, temp_kf)
    return smooth_states

def smooth_all_tracks(pred_tracks):
    """
    pred_tracks: {tid: {frame: (x,y,w,h,is_v(可选),...)}}
    => 返回 smoothed_tracks: {tid: {frame: (x_s,y_s,w,h,vx,vy,...) } }
    """
    smoothed_tracks = {}
    for tid, frames_dict in pred_tracks.items():
        if len(frames_dict)==0:
            smoothed_tracks[tid] = {}
            continue
        
        sorted_frames = sorted(frames_dict.keys())
        # 收集 measurements
        meas_list = []
        for f in sorted_frames:
            (x,y,w,h,*rest) = frames_dict[f]
            meas_list.append((x,y))
        
        # 离线KF + RTS
        smooth_states = offline_smooth_track(meas_list)
        # smooth_states[i] => [x_s,y_s,vx_s,vy_s]
        
        # 合并回
        out_dict = {}
        for i, f in enumerate(sorted_frames):
            (orig_x,orig_y, w,h,*rest) = frames_dict[f]
            x_s = smooth_states[i][0]
            y_s = smooth_states[i][1]
            vx  = smooth_states[i][2]
            vy  = smooth_states[i][3]
            # is_virtual? (index 4?), 其余信息自选
            is_v = False
            if len(rest)>0 and isinstance(rest[0], bool):
                is_v = rest[0]
            out_dict[f] = (x_s, y_s, w, h, is_v, vx, vy)
        smoothed_tracks[tid] = out_dict
    return smoothed_tracks

def solve_ilp_iterative(G, cells_by_t, optim_config):
    import pulp
    delta_p_max= optim_config.get('delta_p_max',20.0)
    max_virtual= optim_config.get('max_virtual',5)
    
    max_iter= 10

   
def convert_bbox_to_z(bbox):
    """
    bbox: [x1,y1,x2,y2]
    output: [x, y, s, r]
    """
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w/2.
    y_c = y1 + h/2.
    s = w*h
    r = w/float(h+1e-6)
    return np.array([x_c,y_c,s,r],dtype=np.float32).reshape((4,1))

def convert_x_to_bbox(x):
    """
    x: [x, y, s, r]
    return: [x1,y1,x2,y2]
    """
    x_c,y_c,s,r = x[0], x[1], x[2], x[3]
    w = np.sqrt(s*r)
    h = s/(w+1e-6)
    x1 = x_c - w/2.
    y1 = y_c - h/2.
    x2 = x_c + w/2.
    y2 = y_c + h/2.
    return np.array([x1,y1,x2,y2],dtype=np.float32).reshape((4,))

class KalmanBoxTracker:
    """
    类似 SORT 的卡尔曼滤波器
    """
    count = 0
    def __init__(self, bbox):
        # bbox: [x1,y1,x2,y2]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]], dtype=np.float32)
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]], dtype=np.float32)

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = np.squeeze(convert_bbox_to_z(bbox))
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.age = 0

    def predict(self):
        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] = 0.
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def update(self, bbox):
        self.time_since_update = 0
        z = convert_bbox_to_z(bbox)
        self.kf.update(z)

    def get_state(self):
        """
        返回当前预测框 [x1,y1,x2,y2]
        """
        return convert_x_to_bbox(self.kf.x)

    def get_velocity(self):
        """
        这里假设 x[4], x[5] 近似 vx, vy (像素/帧), 
        具体看F矩阵定义, 仅作演示.
        """
        vx = float(self.kf.x[4])
        vy = float(self.kf.x[5])
        return vx, vy

def iou_1v1(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0., xB - xA)
    interH = max(0., yB - yA)
    interArea = interW * interH
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union>0 else 0.0

def box_from_xywh(x, y, w, h):
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1,y1,x2,y2]

def replay_track_with_kf(frames_map):
    """
    给定一条轨迹(按frame索引的字典: frame -> (x,y,w,h,is_v)),
    从第一个帧到最后一个帧, 依次 predict->update,
    返回 (kf_tracker, frames_map_out)

    frames_map_out: {frame: (x,y,w,h,is_v, vx, vy)}
      其中 x,y,w,h 可能被 KF 矫正, vx,vy 来源于 KF
      也可只存 vx,vy, 不改 x,y,w,h, 灵活处理.
    """
    frames_sorted = sorted(frames_map.keys())
    if not frames_sorted:
        return None, {}

    # 初始化
    f1 = frames_sorted[0]
    (x1,y1,w1,h1,isv1) = frames_map[f1]
    box1 = box_from_xywh(x1,y1,w1,h1)
    kf_tracker = KalmanBoxTracker(box1)

    frames_map_out = {}
    
    # 第一个帧: 先predict(其实只改变 age ),再update
    kf_tracker.predict()
    kf_tracker.update(np.array(box1,dtype=np.float32))
    vx, vy = kf_tracker.get_velocity()
    # 这里我们是否要把KF矫正后的x,y,w,h写回? 视需求而定
    x1_est, y1_est, x2_est, y2_est = kf_tracker.get_state()
    w_est = x2_est - x1_est
    h_est = y2_est - y1_est
    frames_map_out[f1] = (x1_est, y1_est, w_est, h_est, isv1, vx, vy)

    # 从第二帧开始
    for f in frames_sorted[1:]:
        (xx,yy,ww,hh,isv) = frames_map[f]
        box_ = box_from_xywh(xx,yy,ww,hh)

        pred_box = kf_tracker.predict()  # 预测
        kf_tracker.update(np.array(box_, dtype=np.float32))  # 更新

        vx, vy = kf_tracker.get_velocity()
        x1_est, y1_est, x2_est, y2_est = kf_tracker.get_state()
        w_est = x2_est - x1_est
        h_est = y2_est - y1_est
        frames_map_out[f] = (x1_est, y1_est, w_est, h_est, isv, vx, vy)

    return kf_tracker, frames_map_out

def to_numpy(data):
    if hasattr(data, 'cpu'): # 如果是 PyTorch Tensor
        return data.cpu().numpy()
    elif isinstance(data, list): # 如果是 Python list，转换为 NumPy 数组
        return np.array(data)
    elif isinstance(data, np.ndarray): # 如果已经是 NumPy 数组
        return data
    else:
        return None # 无法处理的类型

def parse_all_ground_truths(base_path="data/viso/test"):
    """
    读取 base_path 下每个视频 <vid>/gt/gt.txt
    返回:
      video_gt_dict = {
        '001': {
           1: [(obj_id, x_min, y_min, w, h, conf, vx, vy, ...), ... ],
           2: [...],
           ...
        },
        '016': {...},
        ...
      }
      first_appear_dict = {
        ('001', obj_id): frame_id,
        ...
      }
    """
    video_gt_dict = {}
    first_appear_dict = {}
    
    # 列出 test下所有子目录(每个视频)
    videos = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for vid in videos:
        gt_path = os.path.join(base_path, vid, "gt", "gt.txt")
        if not os.path.exists(gt_path):
            continue
        
        video_gt_dict[vid] = defaultdict(list)
        
        with open(gt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    # 不够字段 => 跳过或报错
                    continue
                frame_id = int(parts[0])    # 可能1-based
                obj_id   = int(parts[1])
                x_min    = float(parts[2])
                y_min    = float(parts[3])
                w        = float(parts[4])
                h        = float(parts[5])
                conf     = float(parts[6])
                vx       = float(parts[7])
                vy       = float(parts[8])
                # 若有更多字段，可继续
                tup = (obj_id, x_min, y_min, w, h, conf, vx, vy)
                video_gt_dict[vid][frame_id].append(tup)
                
                key = (vid, obj_id)
                if key not in first_appear_dict:
                    first_appear_dict[key] = frame_id  # 记录首次出现帧
    
    return video_gt_dict, first_appear_dict

def overlap_exceed_10pct(det_box, gt_box):
    """
    det_box: (x_min, y_min, w, h)
    gt_box : (x_min, y_min, w, h)
    若相交面积 >= 0.1 * det_area => True
    """
    (dxmin, dymin, dw, dh) = det_box
    (gxmin, gymin, gw, gh) = gt_box
    
    if dw<=0 or dh<=0:
        return False
    det_area = dw*dh
    
    # intersection
    d_xmax = dxmin + dw
    d_ymax = dymin + dh
    g_xmax = gxmin + gw
    g_ymax = gymin + gh
    
    inter_xmin = max(dxmin, gxmin)
    inter_ymin = max(dymin, gymin)
    inter_xmax = min(d_xmax, g_xmax)
    inter_ymax = min(d_ymax, g_ymax)
    
    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w*inter_h
    
    if inter_area >= 0.1*det_area:
        return True
    return False

def parse_predictions(solution_edges):
    """
    根据选中的边解析轨迹
    返回: pred_tracks (track_id -> {frame: (x, y, w, h)}),
           frame_tracks (frame -> list of (track_id, x, y, w, h))
    """
    node_succ = {}
    node_pred = {}
    node_pos = {}
    for u, v, posu, posv in solution_edges:
        node_succ[u] = v
        node_pred[v] = u
        node_pos[u] = posu
        node_pos[v] = posv

    # 找起始节点（无前驱）
    all_nodes = set(node_succ.keys()).union(set(node_succ.values()))
    start_nodes = [nd for nd in all_nodes if nd not in node_pred]

    pred_tracks = {}
    frame_tracks = {}
    track_id = 0
    visited = set()

    for st in start_nodes:
        if st in visited:
            continue
        tid = track_id
        track_id += 1
        pred_tracks[tid] = {}
        node = st
        while node in node_succ:
            if node in visited:
                break
            visited.add(node)
            if node in node_pos:
                fr, cell_id = node
                px, py, w, h = node_pos[node]
                pred_tracks[tid][fr] = (px, py, w, h)
                if fr not in frame_tracks:
                    frame_tracks[fr] = []
                frame_tracks[fr].append((tid, px, py, w, h))
            node = node_succ[node]
        # 末尾节点
        if node not in visited and node in node_pos:
            visited.add(node)
            fr, cell_id = node
            px, py, w, h = node_pos[node]
            pred_tracks[tid][fr] = (px, py, w, h)
            if fr not in frame_tracks:
                frame_tracks[fr] = []
            frame_tracks[fr].append((tid, px, py, w, h))

    return pred_tracks, frame_tracks

# ==========================================
# 1. 核心数据结构：Tracklet 提取
# ==========================================

def extract_tracklets_from_mot(mot_data):
    """
    将原始 MOT 列表转换为函数期待的字典结构
    输入 mot_data: [[f, id, x, y, w, h, ...], ...]
    """
    if not mot_data or len(mot_data) == 0:
        return []
        
    # 使用 dict 按 ID 聚合路径点
    tracks_dict = defaultdict(lambda: {"frames": [], "bboxes": []})
    
    for row in mot_data:
        # 根据你的 DEBUG 输出: [1, 1, 93.02, 253.11, 4.89, 4.72, ...]
        f = int(row[0])
        tid = int(row[1])
        x, y, w, h = row[2:6]
        
        tracks_dict[tid]["frames"].append(f)
        # 转换为 [x1, y1, x2, y2] 用于后续 IoU 计算
        tracks_dict[tid]["bboxes"].append([x, y, x + w, y + h])
    
    # 必须返回 .values() 的列表形式
    return list(tracks_dict.values())

# ==========================================
# 2. 全局优化逻辑：片段连接约束
# ==========================================

def convert_tracklets_to_internal_edges(tracklets, cells_by_t, config, iou_thresh=0.5):
    """
    逻辑核心：根据时空约束，将 Tracklets 转换为 ILP 图中的有向边。
    由于你不需要道路约束了，这里主要关注：
    1. 时间先后顺序 (A结束 < B开始)
    2. 运动平滑度 (卡尔曼滤波预测或线性外推)
    """
    edges = []
    tracklet_ids = list(tracklets.keys())
    
    for i in range(len(tracklet_ids)):
        for j in range(len(tracklet_ids)):
            if i == j: continue
            
            A = tracklets[tracklet_ids[i]]
            B = tracklets[tracklet_ids[j]]
            
            # 时间约束：A 必须在 B 之前结束，且间隔不能太长（如不超过 30 帧）
            time_gap = B["start_frame"] - A["end_frame"]
            if 0 < time_gap <= config.get("max_gap", 30):
                
                # 运动预测约束：
                # 假设 A 以最后两帧的速度匀速运动，预测它在 B 开始时刻的位置
                last_box_A = np.array(A["bboxes"][-1])
                first_box_B = np.array(B["bboxes"][0])
                
                # 计算空间距离或预测 IOU
                dist = np.linalg.norm(last_box_A[:2] - first_box_B[:2])
                
                # 如果距离在合理范围内，则认为可能连接
                if dist < config.get("max_dist", 100):
                    # 计算边权（负对数似然）
                    # 权重越小，ILP 越倾向于连接
                    weight = dist * config.get("dist_weight", 1.0) + time_gap * config.get("time_weight", 0.5)
                    edges.append({
                        "from": tracklet_ids[i],
                        "to": tracklet_ids[j],
                        "weight": weight
                    })
    return edges

def diagnose_match(structured_tracklets, cells_by_t, iou_thresh=0.25, scale_factor=None):
    print("--- 开始诊断匹配逻辑 ---")
    
    # 1. 检查 Cell 转换结果
    cells_xyxy = defaultdict(dict)
    sample_t = None
    for t, cell_list in cells_by_t.items():
        sample_t = t
        for cell_tuple in cell_list:
            cid = cell_tuple[0]
            pos = cell_tuple[1] # [cx, cy, w, h]
            # 这里的转换逻辑必须与数据来源一致
            cells_xyxy[t][cid] = np.array([
                pos[0] - pos[2]/2, pos[1] - pos[3]/2,
                pos[0] + pos[2]/2, pos[1] + pos[3]/2
            ])
        if len(cells_xyxy) > 0: break # 只检查第一帧

    # 2. 打印第一帧对比
    if sample_t is not None and len(structured_tracklets) > 0:
        track = structured_tracklets[0]
        if sample_t in track['frames']:
            idx = track['frames'].index(sample_t)
            raw_ft_box = np.array(track['bboxes'][idx])
            processed_ft_box = raw_ft_box * scale_factor if scale_factor else raw_ft_box
            
            print(f"帧号: {sample_t}")
            print(f"轨迹原始框 (FT): {raw_ft_box}")
            print(f"缩放后框 (FT*scale): {processed_ft_box}")
            
            print(f"\n当前帧共有 {len(cells_xyxy[sample_t])} 个 Cell 节点")
            max_debug_iou = 0
            for cid, cell_box in cells_xyxy[sample_t].items():
                iou = calculate_iou_xyxy_numpy(processed_ft_box, cell_box)
                if iou > max_debug_iou: max_debug_iou = iou
            
            print(f"本帧最高 IoU: {max_debug_iou:.4f} (阈值要求: {iou_thresh})")
            
            if max_debug_iou == 0:
                # 进一步分析原因
                cell_sample = list(cells_xyxy[sample_t].values())[0]
                print(f"\n[分析] 坐标范围不匹配:")
                print(f"FT 框中心: {((processed_ft_box[0]+processed_ft_box[2])/2, (processed_ft_box[1]+processed_ft_box[3])/2)}")
                print(f"Cell 框中心 (第一个): {((cell_sample[0]+cell_sample[2])/2, (cell_sample[1]+cell_sample[3])/2)}")
                
                dist = np.sqrt(((processed_ft_box[0]-cell_sample[0])**2 + (processed_ft_box[1]-cell_sample[1])**2))
                print(f"中心点距离: {dist:.2f} 像素")
                
    print("--- 诊断结束 ---")

def analyze_correspondence(structured_tracklets, cells_data, scale_factor):
    """
    分析 FT 轨迹与 Cell 节点是否具有相同的运动轨迹指纹
    修复了 ValueError: The truth value of an array... 错误
    """
    import numpy as np
    results = []

    for track_idx, track in enumerate(structured_tracklets):
        frames = track['frames']
        bboxes = np.array(track['bboxes'])
        
        # FT 轨迹中心点 (在原始分辨率空间)
        ft_centers = np.zeros((len(bboxes), 2))
        ft_centers[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
        ft_centers[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
        
        offsets = []
        for i, f_num in enumerate(frames):
            curr_cells = cells_data.get(f_num, [])
            if not curr_cells:
                continue
            
            try:
                # 提取 Y 坐标并强制转换为标量 float
                cell_ys = []
                for c in curr_cells:
                    if isinstance(c, dict):
                        val = c['center'][1]
                    else:
                        val = c[1]
                    
                    # 关键修复：如果是 numpy 数组，取其第一个元素或转换为标量
                    if isinstance(val, (np.ndarray, list)):
                        cell_ys.append(float(val[0]))
                    else:
                        cell_ys.append(float(val))
                
                if not cell_ys:
                    continue

                # FT 的 Y 坐标乘以缩放因子
                ft_y_scaled = float(ft_centers[i, 1] * scale_factor)
                
                # 计算差值的绝对值，确保是在标量之间计算
                diffs = [abs(cy - ft_y_scaled) for cy in cell_ys]
                min_offset = min(diffs)
                offsets.append(min_offset)
                
            except Exception as e:
                # 打印错误以防万一，但跳过该帧处理
                print(f"Error processing frame {f_num}: {e}")
                continue

        if offsets:
            std_dev = np.std(offsets)
            avg_offset = np.mean(offsets)
            results.append({
                "track_id": track_idx,
                "y_offset_avg": avg_offset,
                "y_offset_std": std_dev,
                "status": "Match" if std_dev < 5.0 else "Mismatched"
            })
            # 这里的打印可以帮你确定 scale_factor 是否准确
            if track_idx % 10 == 0: # 减少日志量
                print(f"Track {track_idx}: Avg_Y_Offset={avg_offset:.2f}, Std={std_dev:.2f}")

    return results

class AlignmentDebugger:
    """
    专门用于诊断 FT 轨迹与 Graph Cell 空间对齐问题的调试工具
    """
    def __init__(self, scale_factor=0.4128, offset_y=0, offset_x=0):
        self.scale = scale_factor
        self.oy = offset_y
        self.ox = offset_x

    def visualize_alignment(self, frame_img, ft_boxes, cell_nodes, frame_idx):
        """
        在图像上绘制 FT 框(红色)和 Cell 框(绿色)，观察空间偏移
        frame_img: 原始图像或空画布
        ft_boxes: FT 原始坐标 [[x1, y1, x2, y2], ...]
        cell_nodes: Graph 中的 Cell 节点 [[cx, cy, w, h], ...]
        """
        # 如果没有图，创建一个黑色画布用于观察相对位置
        if frame_img is None:
            canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
        else:
            canvas = frame_img.copy()

        # 1. 绘制 Cell 节点 (绿色) - 假设这是基准坐标系
        for node in cell_nodes:
            cx, cy, w, h = node
            x1, y1 = int(cx - w/2), int(cy - h/2)
            x2, y2 = int(cx + w/2), int(cy + h/2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, "Cell", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 2. 绘制原始 FT 框 (黄色) - 观察原始位置
        for box in ft_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)

        # 3. 绘制应用缩放和偏移后的 FT 框 (红色) - 观察是否与绿色重合
        for box in ft_boxes:
            nx1 = int(box[0] * self.scale + self.ox)
            ny1 = int(box[1] * self.scale + self.oy)
            nx2 = int(box[2] * self.scale + self.ox)
            ny2 = int(box[3] * self.scale + self.oy)
            cv2.rectangle(canvas, (nx1, ny1), (nx2, ny2), (0, 0, 255), 2)
            cv2.putText(canvas, "FT_Adj", (nx1, ny1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.title(f"Frame {frame_idx}: Green=Cell, Red=FT_Adjusted, Yellow=FT_Raw")
        plt.show()

    def print_diagnostic_report(self, ft_boxes, cell_nodes):
        """
        计算统计意义上的建议偏置
        ft_boxes: [[x1, y1, x2, y2], ...]
        cell_nodes: [[cx, cy, w, h], ...]
        """
        if not ft_boxes or not cell_nodes:
            print("[DEBUG] 当前帧样本数据不足，无法进行偏移量诊断。")
            return

        # 计算中心点
        ft_centers = np.array([[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in ft_boxes])
        cell_centers = np.array([[c[0], c[1]] for c in cell_nodes])

        # 应用缩放
        scaled_ft = ft_centers * self.scale
        
        offsets = []
        for s_ft in scaled_ft:
            # 寻找距离最近的 Cell 节点
            dists = np.linalg.norm(cell_centers - s_ft, axis=1)
            min_idx = np.argmin(dists)
            # 150像素内认为是可能的同一个目标（根据缩放后的尺度衡量）
            if dists[min_idx] < 150: 
                offsets.append(cell_centers[min_idx] - s_ft)
        
        if offsets:
            avg_offset = np.mean(offsets, axis=0)
            print(f"\n" + "="*40)
            print(f"--- 自动对齐诊断报告 ---")
            print(f"当前使用的 Scale: {self.scale}")
            print(f"检测到平均 X 偏移 (ox): {avg_offset[0]:.2f}")
            print(f"检测到平均 Y 偏移 (oy): {avg_offset[1]:.2f}")
            print(f"建议代码修改建议:")
            print(f"nx1 = int(box[0] * {self.scale} + {avg_offset[0]:.2f})")
            print(f"ny1 = int(box[1] * {self.scale} + {avg_offset[1]:.2f})")
            print("="*40 + "\n")
        else:
            print("[DEBUG] 未能匹配到足够近的节点，请检查 scale_factor 是否偏差过大。")
            
def calculate_iou_xyxy_numpy(box1, box2):
    """ 计算两个 [x1, y1, x2, y2] 盒子的 IoU """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0

def match_tracklets_to_nodes(structured_tracklets, cells_by_t, iou_thresh=0.1, scale_factor=0.4128, y_offset=100.0):
    """
    将 FastTracker 的轨迹匹配到图节点上。
    
    Args:
        y_offset: 根据诊断结果，FT 的 Y 坐标需要加上这个偏移才能对齐 Cell 节点。
    """
    fixed_edges = []
    
    # 将 FT 轨迹转换为 (frame, id) 到 node_id 的映射
    # 方便查找连续帧之间的边
    ft_to_cell_map = {}

    for track_id, boxes in structured_tracklets.items():
        for i in range(len(boxes)):
            f, x1, y1, x2, y2 = boxes[i]
            
            # 1. 坐标对齐 (缩放 + 偏移)
            # 根据诊断日志：X 轴很准，Y 轴偏了约 100
            adj_box = [
                x1 * scale_factor, 
                y1 * scale_factor + y_offset, 
                x2 * scale_factor, 
                y2 * scale_factor + y_offset
            ]
            
            best_iou = -1
            best_node_id = None
            
            if f in cells_by_t:
                for cell in cells_by_t[f]:
                    cell_id, c_pos = cell[0], cell[1]
                    # Cell 格式通常是 [cx, cy, w, h]
                    cb = [
                        c_pos[0] - c_pos[2]/2, 
                        c_pos[1] - c_pos[3]/2, 
                        c_pos[0] + c_pos[2]/2, 
                        c_pos[1] + c_pos[3]/2
                    ]
                    
                    iou = calculate_iou_xyxy(adj_box, cb)
                    if iou > best_iou:
                        best_iou = iou
                        best_node_id = cell_id
            
            # 只要有基本的重合或距离非常近，就强制绑定
            if best_node_id is not None and best_iou > iou_thresh:
                ft_to_cell_map[(f, track_id)] = best_node_id

        # 2. 生成边：如果连续两帧的 FT 点都匹配到了 Cell 节点，则生成一条固定边
        for i in range(len(boxes) - 1):
            f_curr, _, _, _, _ = boxes[i]
            f_next, _, _, _, _ = boxes[i+1]
            
            u_key = (f_curr, track_id)
            v_key = (f_next, track_id)
            
            if u_key in ft_to_cell_map and v_key in ft_to_cell_map:
                u = (f_curr, ft_to_cell_map[u_key])
                v = (f_next, ft_to_cell_map[v_key])
                # 给一个极高的权重，确保 ILP 喜欢它，或者后面直接强制约束
                fixed_edges.append((u, v, 5000.0))

    print(f"[MATCH INFO] 成功匹配点: {len(ft_to_cell_map)}, 生成固定边: {len(fixed_edges)}")
    return fixed_edges

def calculate_iou_box(box1, box2):
    """ 计算两个 [x1, y1, x2, y2] 的 IoU """
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])
    
    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

def calculate_iou_correct(box_a, box_b, format='xyxy'):
    """
    box_a: pred [x1, y1, x2, y2] 或 [x1, y1, w, h]
    box_b: gt   [x1, y1, x2, y2]
    """
    if format == 'tlwh':
        box_a = [box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3]]
    
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def debug_visualize_match(video_id, frame_id, pred_boxes, gt_boxes, img_dir):
    """
    可视化特定帧的预测和GT，定位为何匹配失败
    """
    # 尝试寻找图像路径
    img_path = os.path.join(img_dir, video_id, f"{frame_id:06d}.jpg")
    if not os.path.exists(img_path):
        # 尝试另一种命名格式
        img_path = os.path.join(img_dir, video_id, f"{frame_id}.jpg")
        
    if os.path.exists(img_path):
        canvas = cv2.imread(img_path)
        # 画 GT (绿色)
        for g in gt_boxes:
            # g: (obj_id, x, y, w, h)
            cv2.rectangle(canvas, (int(g[1]), int(g[2])), (int(g[1]+g[3]), int(g[2]+g[4])), (0, 255, 0), 2)
            cv2.putText(canvas, f"GT_{int(g[0])}", (int(g[1]), int(g[2])-5), 1, 1, (0, 255, 0), 1)
            
        # 画 Pred (红色)
        for p in pred_boxes:
            # p: [x1, y1, w, h] 假设你存入的是 tlwh
            cv2.rectangle(canvas, (int(p[0]), int(p[1])), (int(p[0]+p[2]), int(p[1]+p[3])), (0, 0, 255), 2)
            cv2.putText(canvas, f"P_{int(p[5])}", (int(p[0]), int(p[1])-5), 1, 1, (0, 0, 255), 1)
            
        cv2.imshow("Debug Match (Green:GT, Red:Pred)", canvas)
        cv2.waitKey(0) # 按任意键继续

def analyze_trajectory_quality_v2(all_video_tracklets, video_gt_dict, img_base_dir=None):
    print("\n" + "="*20 + " 轨迹质量分析修正版 " + "="*20)
    
    total_tracks = 0
    matched_tracks = 0
    
    for video_id, flat_tracklets in all_video_tracklets.items():
        if not flat_tracklets: continue
        
        tracks = defaultdict(list)
        for item in flat_tracklets:
            tid = item[1]
            tracks[tid].append(item)
            
        gt_data = video_gt_dict.get(video_id, {})
        if not gt_data:
            print(f"警告: 视频 {video_id} 没有对应的 GT 数据!")
            continue

        for tid, points in tracks.items():
            total_tracks += 1
            match_hits = 0
            
            for p in points:
                fid = int(p[0])
                # 【重要修正】：如果 baseline_video_data 存的是 tlwh，这里必须对应处理
                # p 的索引: 0:frame, 1:id, 2:x1, 3:y1, 4:w, 5:h, 6:conf
                pred_tlwh = [p[2], p[3], p[4], p[5]]
                
                gts = gt_data.get(fid, [])
                best_iou = 0
                for gt in gts:
                    # GT 通常是 (id, x, y, w, h)
                    gt_xyxy = [gt[1], gt[2], gt[1]+gt[3], gt[2]+gt[4]]
                    iou = calculate_iou_correct(pred_tlwh, gt_xyxy, format='tlwh')
                    if iou > best_iou:
                        best_iou = iou
                
                if best_iou > 0.3: # 调试阶段可以先调低阈值看有没有任何匹配
                    match_hits += 1
                
                # 调试代码：如果是前几条轨迹，且没匹配上，开启可视化
                if total_tracks < 5 and best_iou < 0.1 and img_base_dir:
                    print(f"调试可视化: 视频 {video_id} 帧 {fid} 匹配失败, Best IoU: {best_iou:.4f}")
                    # 重新传入该帧所有预测供对比
                    current_frame_preds = [ [it[2], it[3], it[4], it[5], it[6], it[1]] for it in flat_tracklets if int(it[0]) == fid]
                    debug_visualize_match(video_id, fid, current_frame_preds, gts, img_base_dir)

            if len(points) > 0 and (match_hits / len(points) > 0.5):
                matched_tracks += 1

    print(f"分析完成: TP={matched_tracks}, FP={total_tracks - matched_tracks}")

def calculate_iou_xyxy(boxA, boxB):
    """计算两个 [x1, y1, x2, y2] 格式框的 IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight
    
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = float(areaA + areaB - interArea)
    return interArea / unionArea if unionArea > 0 else 0

def filter_tracklets_by_gt(initial_tracklets, current_gt, iou_thresh=0.5):
    """
    修正后的函数：适配 (frame, id, x1, y1, x2, y2, conf, ...) 格式
    """
    cleaned_tracklets = []
    total_count = 0
    kept_count = 0

    # 1. 容错处理：如果当前视频没有检测结果
    if not initial_tracklets:
        return [], {'total': 0, 'kept': 0}

    for det in initial_tracklets:
        total_count += 1
        
        # 根据 DEBUG 内容：
        # det[0]=frame, det[1]=id, det[2]=x1, det[3]=y1, det[4]=x2, det[5]=y2
        try:
            frame_id = int(det[0])
            pred_bbox = [float(det[2]), float(det[3]), float(det[4]), float(det[5])]
        except (IndexError, TypeError):
            continue
            
        gt_list = current_gt.get(frame_id, [])
        is_correct = False
        
        for gt_tup in gt_list:
            # 假设 GT 格式是 (obj_id, x_min, y_min, w, h, ...)
            # 需要把 GT 也转换成 xyxy 参与计算
            gt_x1, gt_y1 = gt_tup[1], gt_tup[2]
            gt_x2, gt_y2 = gt_tup[1] + gt_tup[3], gt_tup[2] + gt_tup[4]
            gt_bbox = [gt_x1, gt_y1, gt_x2, gt_y2]
            
            iou = calculate_iou_xyxy(pred_bbox, gt_bbox)
            
            if iou >= iou_thresh:
                is_correct = True
                break
        
        if is_correct:
            cleaned_tracklets.append(det)
            kept_count += 1

    stats = {'total': total_count, 'kept': kept_count}
    return cleaned_tracklets, stats


def create_graph(cells_by_t, optim_config, road_mask = None, tracklet_edges=None):
    

    G = nx.DiGraph()

    potential_edges = []

    # 从优化配置中获取参数
    max_children = optim_config.get('max_children', 3)
    distance_threshold = optim_config.get('distance_threshold', 10)
    confidence_threshold = optim_config.get('confidence_threshold', 0.3)
    confidence_function = optim_config.get('confidence_function', 'quadratic')
    weight_position = optim_config.get('weight_position', 1.0)
    weight_velocity = optim_config.get('weight_velocity', 1.0)
    weight_appearance = optim_config.get('weight_appearance', 1.0)
    epsilon = 1
    gaussian_sigma = optim_config.get('gaussian_sigma', 1.0)

    # 只考虑相邻帧
    time_window = optim_config.get('time_window', 5)

    # 道路约束强度：1.0表示不惩罚，<1.0表示降低不在路上目标的权重
    road_penalty_factor = optim_config.get('road_penalty_factor', 0.5)

    # 1. 添加节点 (保持原逻辑，过滤低分点)
    for t in cells_by_t:
        for cell in cells_by_t[t]:
            cell_id, position = cell[0], cell[1]
            # 统一提取特征：(id, pos, vel, conf, app)
            conf = cell[3] if len(cell) > 3 else 1.0
            # # --- 道路约束应用 (节点级别) ---
            # cx, cy = int(position[0]), int(position[1])
            # is_on_road = True
            # if 0 <= cy < road_mask.shape[0] and 0 <= cx < road_mask.shape[1]:
            #     if road_mask[cy, cx] == 0:
            #         is_on_road = False
            
            # # 如果点不在路上，大幅降低其有效置信度
            # effective_conf = conf if is_on_road else conf * road_penalty_factor
            
            # if effective_conf < confidence_threshold:
            #     continue
            
            vel = cell[2] if len(cell) > 2 else np.array([0, 0])
            app = cell[4] if len(cell) > 4 else None
            
            G.add_node((t, cell_id), pos=position, velocity=vel, confidence=conf, appearance=app)

    # 2. 跨帧建边逻辑
    frames = sorted(cells_by_t.keys())
    for idx, t in enumerate(frames):
        # 查找当前帧在图中的活跃节点
        current_nodes = [n for n in G.nodes() if n[0] == t]
        if not current_nodes: continue

        # 向后搜索 time_window 范围内的帧
        for gap in range(1, time_window + 1):
            if idx + gap >= len(frames): break
            t_next = frames[idx + gap]
            
            next_nodes = [n for n in G.nodes() if n[0] == t_next]
            if not next_nodes: continue

            # 构建 KDTree 加速空间搜索
            next_positions = [G.nodes[n]['pos'][:2] for n in next_nodes]
            kd_tree = KDTree(next_positions)

            for u in current_nodes:
                u_data = G.nodes[u]
                # 预测位置：当前位置 + 速度 * 时间间隔
                predicted_pos = u_data['pos'][:2] + u_data['velocity'][:2] * gap
                
                distances, indices = kd_tree.query(predicted_pos, k=min(max_children, len(next_nodes)))
                if isinstance(distances, (float, int)):
                    distances, indices = [distances], [indices]

                for dist, index in zip(distances, indices):
                    if dist < distance_threshold:
                        v = next_nodes[index]
                        v_data = G.nodes[v]

                        # --- 核心权重计算优化 ---
                        # 1. 位置相似度 (考虑预测偏差)
                        sim_p = np.exp(-(dist ** 2) / (2 * (gaussian_sigma * gap) ** 2))
                        
                        # 2. 速度方向相似度
                        delta_v = np.linalg.norm(u_data['velocity'] - v_data['velocity'])
                        sim_v = np.exp(-(delta_v ** 2) / (2 * gaussian_sigma ** 2))

                        # 3. 外观相似度
                        sim_app = 0.0
                        if weight_appearance > 0 and u_data['appearance'] is not None and v_data['appearance'] is not None:
                            feat_u = to_numpy(u_data['appearance']).flatten().reshape(1, -1)
                            feat_v = to_numpy(v_data['appearance']).flatten().reshape(1, -1)
                            sim_app = cosine_similarity(feat_u, feat_v)[0][0]


                        # # --- 道路约束应用 (边级别) ---
                        # # 如果连接的两个点中有一个不在路上，降低这条边的连接权重
                        # edge_road_factor = 1.0
                        # if not u_data['on_road'] or not v_data['on_road']:
                        #     edge_road_factor = road_penalty_factor

                        # 4. 置信度调节 (新增)
                        # 综合两点置信度，高分连接具有更高基础权重
                        conf_factor = (u_data['confidence'] * v_data['confidence']) ** 0.5
                        
                        # 5. 时间惩罚 (跨帧越多，奖励衰减越多)
                        time_decay = 0.9 ** (gap - 1)

                        # 最终奖励计算
                        # reward = conf_factor * time_decay* edge_road_factor * (
                        reward = conf_factor * time_decay * (
                            weight_position * sim_p + 
                            weight_velocity * sim_v + 
                            weight_appearance * sim_app
                        )
                        
                        potential_edges.append((u, v, reward))

    # 3. 最终边入图逻辑 (保持原逻辑：ILP通常处理成本即 -reward)
    tracklet_edge_map = {(u, v): w for u, v, w in tracklet_edges} if tracklet_edges else {}

    rewards = [e[2] for e in potential_edges]
    if rewards:
        print(f"DEBUG: Reward Max: {max(rewards)}, Min: {min(rewards)}, Mean: {np.mean(rewards)}")
        print(f"DEBUG: Start_Cost: {optim_config.get('start_cost')}")
        print(f"DEBUG: Time_window:{time_window}")
        print(f"DEBUG: Road_penalty_factor:{road_penalty_factor}")

    for u, v, reward in potential_edges:
        # 如果是原有轨迹边，增加 bonus 鼓励保持
        if (u, v) in tracklet_edge_map:
            final_cost = reward + optim_config.get('tracklet_bonus', 1000.0)
        else:
            final_cost = reward
            
        # 注意：ILP solve_ilp 若是 LpMaximize，这里存正值；若是 LpMinimize，存负值
        G.add_edge(u, v, weight=final_cost)

    print(f"DEBUG: Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# def create_graph(cells_by_t, optim_config, road_mask=None, tracklet_edges=None):
#     """
#     构建用于 ILP 的有向图。
#     修复了解包错误 (ValueError: too many values to unpack)。
#     """
#     G = nx.DiGraph()
    
#     # 1. 预处理 FastTracker 边 (处理三元组: u, v, weight)
#     ft_nodes = set()
#     ft_edge_set = set()
#     tracklet_bonus = optim_config.get('tracklet_bonus', 2000.0)

#     if tracklet_edges:
#         for edge in tracklet_edges:
#             # 兼容处理: 无论传入的是 (u,v) 还是 (u,v,w)
#             u, v = edge[0], edge[1]
#             ft_nodes.add(u)
#             ft_nodes.add(v)
#             ft_edge_set.add((u, v))

#     # 参数提取
#     max_children = optim_config.get('max_children', 3)
#     distance_threshold = optim_config.get('distance_threshold', 15)
#     confidence_threshold = optim_config.get('confidence_threshold', 0.2)
#     gaussian_sigma = optim_config.get('gaussian_sigma', 1.0)
#     weight_position = optim_config.get('weight_position', 10.0)
#     weight_velocity = optim_config.get('weight_velocity', 1.0)
#     weight_appearance = optim_config.get('weight_appearance', 1.0)
#     time_window = optim_config.get('time_window', 5)
#     road_penalty_factor = optim_config.get('road_penalty_factor', 0.1)

#     # 2. 添加节点
#     for t, cells in cells_by_t.items():
#         for cell in cells:
#             cell_id, position = cell[0], cell[1]
#             conf = cell[3] if len(cell) > 3 else 1.0
            
#             # 道路检查
#             cx, cy = int(position[0]), int(position[1])
#             is_on_road = True
#             if road_mask is not None:
#                 if 0 <= cy < road_mask.shape[0] and 0 <= cx < road_mask.shape[1]:
#                     if road_mask[cy, cx] == 0: is_on_road = False
            
#             # 强制保留 FT 节点，否则按置信度和道路过滤
#             if (t, cell_id) not in ft_nodes:
#                 effective_conf = conf if is_on_road else conf * road_penalty_factor
#                 if effective_conf < confidence_threshold:
#                     continue
            
#             vel = cell[2] if len(cell) > 2 else np.array([0, 0])
#             app = cell[4] if len(cell) > 4 else None
            
#             G.add_node((t, cell_id), 
#                        pos=position, 
#                        velocity=vel, 
#                        confidence=conf, 
#                        appearance=app, 
#                        on_road=is_on_road)

#     # 3. 跨帧建边逻辑
#     frames = sorted(cells_by_t.keys())
#     for idx, t in enumerate(frames):
#         current_nodes = [n for n in G.nodes() if n[0] == t]
#         if not current_nodes: continue

#         for gap in range(1, time_window + 1):
#             if idx + gap >= len(frames): break
#             t_next = frames[idx + gap]
#             next_nodes = [n for n in G.nodes() if n[0] == t_next]
#             if not next_nodes: continue

#             next_positions = [G.nodes[n]['pos'][:2] for n in next_nodes]
#             kd_tree = KDTree(next_positions)

#             for u in current_nodes:
#                 u_data = G.nodes[u]
#                 pred_pos = u_data['pos'][:2] + u_data['velocity'][:2] * gap
                
#                 dists, indices = kd_tree.query(pred_pos, k=min(max_children, len(next_nodes)))
#                 if isinstance(dists, (float, int)): dists, indices = [dists], [indices]

#                 for dist, index in zip(dists, indices):
#                     if dist < distance_threshold:
#                         v = next_nodes[index]
#                         v_data = G.nodes[v]
                        
#                         # 计算基础奖励
#                         sim_p = np.exp(-(dist ** 2) / (2 * (gaussian_sigma * gap) ** 2))
#                         sim_v = np.exp(-(np.linalg.norm(u_data['velocity'] - v_data['velocity']) ** 2) / (2 * gaussian_sigma ** 2))
#                         sim_app = 0.0
#                         if weight_appearance > 0 and u_data['appearance'] is not None and v_data['appearance'] is not None:
#                             feat_u = to_numpy(u_data['appearance']).flatten().reshape(1, -1)
#                             feat_v = to_numpy(v_data['appearance']).flatten().reshape(1, -1)
#                             sim_app = float(cosine_similarity(feat_u, feat_v)[0][0])

#                         edge_road_factor = 1.0 if (u_data['on_road'] and v_data['on_road']) else road_penalty_factor

#                         reward = (u_data['confidence'] * v_data['confidence'])**0.5 * edge_road_factor * (
#                             weight_position * sim_p + weight_velocity * sim_v + weight_appearance * sim_app
#                         )
                        
#                         # 加上原有轨迹的 Bonus
#                         if (u, v) in ft_edge_set:
#                             reward += tracklet_bonus
                        
#                         G.add_edge(u, v, weight=reward)

#     # 4. 兜底确保 FT 边一定存在于图中 (哪怕超出了 distance_threshold)
#     if ft_edge_set:
#         for u, v in ft_edge_set:
#             if u in G.nodes and v in G.nodes:
#                 if not G.has_edge(u, v):
#                     G.add_edge(u, v, weight=tracklet_bonus)

#     print(f"DEBUG: Graph created. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
#     return G
# def create_refined_graph(cells_by_t, optim_config, tracklet_edges=None):
#     """
#     构建图结构，强化 FastTracker 边的权重。
#     """
#     G = nx.DiGraph()
    
#     # 参数提取
#     max_children = optim_config.get('max_children', 3)
#     distance_threshold = optim_config.get('distance_threshold', 50)
#     time_window = optim_config.get('time_window', 5)
    
#     # 核心：必须赋予 FastTracker 边绝对优先权
#     # 设为一个远大于普通距离得分的值
#     FAST_TRACKER_BONUS = 500.0 

#     # 1. 添加节点
#     for t, cells in cells_by_t.items():
#         for cell in cells:
#             cell_id, position = cell[0], cell[1]
#             conf = cell[3] if len(cell) > 3 else 1.0
#             vel = cell[2] if len(cell) > 2 else np.array([0, 0])
#             app = cell[4] if len(cell) > 4 else None
            
#             G.add_node((t, cell_id), pos=position, velocity=vel, confidence=conf, appearance=app)

#     # 2. 注入 FastTracker 的边 (必须先注入)
#     # 我们认为这些边是“真值”或“锚点”
#     ft_edge_set = set()
#     if tracklet_edges:
#         for (u, v) in tracklet_edges:
#             if u in G.nodes and v in G.nodes:
#                 # 赋予极高权重
#                 G.add_edge(u, v, weight=FAST_TRACKER_BONUS, is_fast_tracker=True)
#                 ft_edge_set.add((u, v))

#     # 3. 构建候选优化边 (用于缝补断裂)
#     frames = sorted(cells_by_t.keys())
#     for idx, t in enumerate(frames):
#         current_nodes = [n for n in G.nodes() if n[0] == t]
        
#         for gap in range(1, time_window + 1):
#             if idx + gap >= len(frames): break
#             t_next = frames[idx + gap]
#             next_nodes = [n for n in G.nodes() if n[0] == t_next]
#             if not next_nodes: continue

#             next_pos_list = [G.nodes[n]['pos'][:2] for n in next_nodes]
#             tree = KDTree(next_pos_list)
            
#             for u in current_nodes:
#                 # 如果节点 u 已经有了 FastTracker 的出边，且 gap=1，
#                 # 我们倾向于不给它寻找额外的候选，除非是为了跨帧补洞
#                 if gap == 1 and any(G.successors(u)):
#                     continue
                    
#                 u_data = G.nodes[u]
#                 pred_pos = u_data['pos'][:2] + u_data['velocity'][:2] * gap
                
#                 dists, indices = tree.query(pred_pos, k=min(max_children, len(next_nodes)))
#                 if isinstance(dists, float): dists, indices = [dists], [indices]
                
#                 for d, i in zip(dists, indices):
#                     v = next_nodes[i]
#                     if (u, v) in ft_edge_set: continue # 已处理
                    
#                     if d < distance_threshold:
#                         # 计算普通运动相似度
#                         # 分数 = 基准分 - 距离惩罚
#                         score = max(0.1, 20.0 - (d / 5.0)) 
#                         # 确保普通边的权重远小于 FAST_TRACKER_BONUS
#                         G.add_edge(u, v, weight=score, is_fast_tracker=False)

#     return G

# def solve_refined_ilp(graph, optim_config):
#     """
#     求解 ILP，增加对轨迹开启的惩罚以减少 FP，同时强制保留高分边。
#     """
#     # 增加 start_cost 惩罚（负值），防止产生大量 1-2 帧的孤立点
#     start_cost = optim_config.get('start_cost', -100.0) 
#     # 节点自身的置信度贡献
#     conf_weight = optim_config.get('conf_weight', 10.0)
    
#     nodes = list(graph.nodes())
#     problem = pulp.LpProblem("Refined_Tracking", pulp.LpMaximize)

#     # 1. 变量定义
#     edges_vars = {e: pulp.LpVariable(f"x_{e[0]}_{e[1]}".replace(" ", ""), cat=pulp.LpBinary) 
#                   for e in graph.edges()}
#     node_vars = {n: pulp.LpVariable(f"z_{n}".replace(" ", ""), cat=pulp.LpBinary) for n in nodes}
#     start_vars = {n: pulp.LpVariable(f"s_{n}".replace(" ", ""), cat=pulp.LpBinary) for n in nodes}
#     end_vars = {n: pulp.LpVariable(f"e_{n}".replace(" ", ""), cat=pulp.LpBinary) for n in nodes}

#     # 2. 目标函数
#     # 边权重 + 节点置信度收益 + 轨迹开启惩罚
#     obj_edges = pulp.lpSum([edges_vars[e] * graph.edges[e]['weight'] for e in graph.edges()])
#     obj_nodes = pulp.lpSum([node_vars[n] * graph.nodes[n].get('confidence', 0) * conf_weight for n in nodes])
#     obj_starts = pulp.lpSum([start_vars[n] * start_cost for n in nodes])
    
#     problem += obj_edges + obj_nodes + obj_starts

#     # 3. 约束条件
#     for n in nodes:
#         in_edges = [edges_vars[(u, n)] for u in graph.predecessors(n)]
#         out_edges = [edges_vars[(n, v)] for v in graph.successors(n)]
        
#         # 入流平衡: sum(in) + start = active
#         problem += pulp.lpSum(in_edges) + start_vars[n] == node_vars[n]
#         # 出流平衡: sum(out) + end = active
#         problem += pulp.lpSum(out_edges) + end_vars[n] == node_vars[n]
        
#         # 强制约束：如果 FastTracker 已经连上的边，只要节点活跃，边就必须选？
#         # 或者通过权重引导即可，这里采用权重引导。

#     # 4. 求解
#     solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=45)
#     problem.solve(solver)

#     if pulp.LpStatus[problem.status] in ["Optimal", "Not Solved"]:
#         selected_edges = []
#         for (u, v), var in edges_vars.items():
#             if var.varValue and var.varValue > 0.5:
#                 selected_edges.append((u, v, graph.nodes[u]['pos'], graph.nodes[v]['pos']))
#         return selected_edges
#     return []
    # # 添加节点
    # for t in cells_by_t:
    #     for cell in cells_by_t[t]:
    #         if len(cell) == 5:
    #             cell_id, position, velocity, confidence, appearance = cell
    #         elif len(cell) == 4:
    #             cell_id, position, velocity, confidence = cell
    #             appearance = None
    #         else:
    #             cell_id, position, velocity = cell[:3]
    #             confidence = 1.0
    #             appearance = None
    #         if confidence < confidence_threshold:
    #             continue
    #         G.add_node((t, cell_id), pos=position, velocity=velocity, confidence=confidence, appearance=appearance)
    
    # # 添加边(相邻帧)
    # frames = sorted(cells_by_t.keys())

    # for idx, t in enumerate(frames):
    #     t_next = t + 1
    #     if t_next in cells_by_t:
    #         current_frame = [cell for cell in cells_by_t[t] if cell[3] >= confidence_threshold]
    #         next_frame = [cell for cell in cells_by_t[t_next] if cell[3] >= confidence_threshold]

    #         next_positions = [cell[1][:2] for cell in next_frame]
    #         next_ids = [cell[0] for cell in next_frame]
    #         next_velocities = [cell[2] for cell in next_frame]
    #         next_confidences = [cell[3] for cell in next_frame]
    #         next_appearances = [cell[4] if len(cell) > 4 else None for cell in next_frame]

    #         if next_positions:
    #             kd_tree = KDTree(next_positions)

    #             for current_cell in current_frame:
    #                 current_id = current_cell[0]
    #                 current_pos = current_cell[1][:2]
    #                 current_vel = current_cell[2]
    #                 current_conf = current_cell[3]
    #                 current_app = current_cell[4] if len(current_cell) > 4 else None

    #                 distances, indices = kd_tree.query(current_pos, k=max_children)
    #                 if not isinstance(distances, np.ndarray):
    #                     distances = [distances]
    #                     indices = [indices]

    #                 for distance, index in zip(distances, indices):
    #                     if distance < distance_threshold:
    #                         next_id = next_ids[index]
    #                         next_pos = next_positions[index]
    #                         next_vel_val = next_velocities[index]
    #                         next_conf_val = next_confidences[index]
    #                         next_app_val = next_appearances[index]

    #                         delta_p = np.linalg.norm(np.array(next_pos) - np.array(current_pos))
    #                         if gaussian_sigma == 0:
    #                             sim_p = 1 / (delta_p + epsilon)
    #                         else:
    #                             sim_p = np.exp(- (delta_p ** 2) / (2 * gaussian_sigma ** 2))

    #                         delta_v = np.linalg.norm(next_vel_val - current_vel)
    #                         if gaussian_sigma == 0:
    #                             sim_v = 1 / (delta_v + epsilon)
    #                         else:
    #                             sim_v = np.exp(- (delta_v ** 2) / (2 * gaussian_sigma ** 2))

    #                         sim_app = 0.0
    #                         if weight_appearance > 0 and current_app is not None and next_app_val is not None:
    #                             # --- 修正后的代码段 ---
                                
    #                             # 确保特征是 NumPy 数组

    #                             current_app_np = to_numpy(current_app)
    #                             next_app_val_np = to_numpy(next_app_val)
                                
    #                             if current_app_np is not None and next_app_val_np is not None:
    #                                 # 确保形状正确 (例如 [1, D])
    #                                 current_app_list = [current_app_np.flatten()]
    #                                 next_app_list = [next_app_val_np.flatten()]
                                    
    #                                 # 重新计算相似度
    #                                 sim_app = cosine_similarity(current_app_list, next_app_list)[0][0]
    #                             else:
    #                                 sim_app = 0.0 # 无法获取有效特征

    #                         if confidence_function == 'linear':
    #                             w_conf = current_conf * next_conf_val
    #                         elif confidence_function == 'quadratic':
    #                             w_conf = (current_conf * next_conf_val) ** 2
    #                         else:
    #                             w_conf = 1.0

    #                         weight = w_conf * (weight_position * sim_p + weight_velocity * sim_v + weight_appearance * sim_app)
    #                         node_u = (t, current_id)
    #                         node_v = (t_next, next_id)
    #                         # 潜在边存储为 (u, v, reward)
    #                         potential_edges.append((node_u, node_v, weight))
    #                         # G.add_edge((t, current_id), (t_next, next_id), weight=weight)
    

    # final_edges = []
    
    # if tracklet_edges:
    #     # 将 Tracklet 边转换为 map，方便查询 {(u, v): reward}
    #     # 注意：tracklet_edges 中的 w 是奖励 (FastTracker's reward)，不是成本
    #     tracklet_edge_map = {(u, v): w for u, v, w in tracklet_edges}
        
    #     for u, v, reward in potential_edges: # potential_edges 中的 w 是我们计算的奖励
    #         if (u, v) in tracklet_edge_map:
    #             # 如果 FastTracker 确认了这条边，我们使用 FastTracker 的高奖励。
    #             # ILP 成本 = -奖励
    #             final_cost = 100 
    #             G.add_edge(u, v, weight=final_cost)
    #         else:
    #             # 否则，使用基于特征相似度计算出的奖励。
    #             # ILP 成本 = -奖励
    #             final_cost = reward
    #             G.add_edge(u, v, weight=final_cost)
    # else:
    #     # 如果没有 Tracklet 边，所有潜在边都按相似度计算成本。
    #     for u, v, reward in potential_edges:
    #         final_cost = -reward
    #         G.add_edge(u, v, weight=final_cost)
    # print("DEBUG: Sample edge weights:", list(G.edges(data='weight'))[:5])
    # return G


def solve_ilp(graph, optim_config):
    """
    修复后的 ILP 求解器
    逻辑说明：
    - 每个节点 n 对应一个二进制变量 z_n，表示该点是否被包含在轨迹中。
    - 目标函数：Maximize sum(Edge_weight * x_uv) + sum(z_n * start_cost)
    """
    start_cost = optim_config.get('start_cost', -1.0) 
    end_cost = optim_config.get('end_cost', 0.0)
    
    # 过滤掉虚拟标识符，只保留物理节点
    nodes = [n for n in graph.nodes() if n not in ['Start', 'End', 'Virtual_Start', 'Virtual_End']]
    
    problem = pulp.LpProblem("TrackForming_Fixed", pulp.LpMaximize)

    # 1. 变量定义
    # 物理边变量
    edges_vars = {}
    for u, v in graph.edges():
        if u in nodes and v in nodes:
            var_name = f"x_{u}_{v}".replace(" ", "")
            edges_vars[(u, v)] = pulp.LpVariable(var_name, cat=pulp.LpBinary)

    # 节点激活变量 (表示该点是否属于任何轨迹)
    node_vars = {}
    # 轨迹起点/终点变量 (用于替代你之前的虚拟边连接逻辑)
    start_vars = {}
    end_vars = {}

    for n in nodes:
        node_vars[n] = pulp.LpVariable(f"z_{n}".replace(" ", ""), cat=pulp.LpBinary)
        start_vars[n] = pulp.LpVariable(f"s_{n}".replace(" ", ""), cat=pulp.LpBinary)
        end_vars[n] = pulp.LpVariable(f"e_{n}".replace(" ", ""), cat=pulp.LpBinary)

    # 2. 目标函数
    # 边收益 + 开启代价 + 结束收益 (通常end_cost为0)
    obj_edges = pulp.lpSum([edges_vars[e] * graph.edges[e]['weight'] for e in edges_vars])
    obj_starts = pulp.lpSum([start_vars[n] * start_cost for n in nodes])
    obj_ends = pulp.lpSum([end_vars[n] * end_cost for n in nodes])
    
    problem += obj_edges + obj_starts + obj_ends

    # 3. 约束条件
    for n in nodes:
        # 入流平衡：来自其他点的边 + 是否作为起点 = 该节点激活状态
        incoming_edges = [edges_vars[(u, n)] for u in graph.predecessors(n) if (u, n) in edges_vars]
        problem += pulp.lpSum(incoming_edges) + start_vars[n] == node_vars[n], f"InFlow_{n}"
        
        # 出流平衡：去往其他点的边 + 是否作为终点 = 该节点激活状态
        outgoing_edges = [edges_vars[(n, v)] for v in graph.successors(n) if (n, v) in edges_vars]
        problem += pulp.lpSum(outgoing_edges) + end_vars[n] == node_vars[n], f"OutFlow_{n}"
        
        # 节点唯一性约束：每个节点激活状态最大为1 (二进制变量天然满足，但明确逻辑)
        problem += node_vars[n] <= 1, f"MaxOcc_{n}"

    # 4. 求解
    # 使用较短的超时时间，防止网格搜索卡死
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30)
    problem.solve(solver)

    if pulp.LpStatus[problem.status] == "Optimal" or problem.status > 0:
        selected_edges = []
        for (u, v), var in edges_vars.items():
            if var.varValue is not None and var.varValue > 0.5:
                selected_edges.append((
                    u, v, 
                    graph.nodes[u]['pos'], 
                    graph.nodes[v]['pos']
                ))
        return selected_edges
    else:
        return []

def prepare_tracklets_from_flat_list(flat_tracklets):
    """
    将 mot_tracker.get_tracklets() 返回的扁平列表转换为 ILP 需要的结构。
    不需要修改 Fasttracker 内部代码。
    
    输入: [(frame_id, tid, x1, y1, x2, y2, conf, feat, vx, vy), ...]
    输出: 
        tracklets_list: List[List[NodeDict]], 每个子列表是一条完整的轨迹段
        node_map: Dict{node_id: NodeDict}, 用于通过 ID 查找原始信息
    """
    tracklets_dict = defaultdict(list)
    node_map = {}
    
    for item in flat_tracklets:
        # 解包数据 (根据您提供的 get_tracklets 返回格式)
        # item: (frame_id, tid, x1, y1, x2, y2, conf, feat, vx, vy)
        if len(item) < 10: continue # 防御性检查
        
        f_id, tid, x1, y1, x2, y2, conf, feat, vx, vy = item
        
        # 创建唯一的节点 ID
        node_id = f"{int(f_id)}_{int(tid)}"
        
        node_obj = {
            'id': node_id,
            'raw_id': tid,
            'frame': int(f_id),
            'tlwh': [x1, y1, x2-x1, y2-y1], # 用于最后保存
            'pos': np.array([x1 + (x2-x1)/2, y1 + (y2-y1)/2]), # 中心点用于计算距离
            'velocity': np.array([vx, vy]),
            'conf': conf,
            'feat': feat
        }
        
        tracklets_dict[tid].append(node_obj)
        node_map[node_id] = node_obj

    # 将字典转为列表，并确保每个轨迹段内部按帧号排序
    tracklets_list = []
    for tid in sorted(tracklets_dict.keys()):
        # 按帧号排序，保证是时序连续的
        t_nodes = sorted(tracklets_dict[tid], key=lambda x: x['frame'])
        tracklets_list.append(t_nodes)
        
    return tracklets_list, node_map

# --- 核心逻辑: 轨迹段级别的 ILP 优化 ---

def solve_tracklet_level_ilp(tracklets, config):
    """
    轨迹段级别的 ILP 优化逻辑。
    修复了 'list' object has no attribute 'flatten' 错误。
    """
    if not tracklets:
        return []

    # 1. 提取元数据
    metadata = {}
    for i, trk in enumerate(tracklets):
        start_node = trk[0]
        end_node = trk[-1]
        
        # 改进特征提取：只取置信度高的帧的特征
        # 增加对 feat 类型的检查，确保能安全调用 flatten
        feats = []
        for n in trk:
            f = n.get('feat')
            conf = n.get('conf', 0)
            if f is not None and conf > 0.4:
                # 处理不同类型的特征输入
                if isinstance(f, torch.Tensor):
                    f_np = f.detach().cpu().numpy().flatten()
                elif isinstance(f, np.ndarray):
                    f_np = f.flatten()
                elif isinstance(f, list):
                    f_np = np.array(f).flatten()
                else:
                    continue
                feats.append(f_np)

        if not feats: # 如果没有高置信度的，就取全部
            for n in trk:
                f = n.get('feat')
                if f is not None:
                    if isinstance(f, torch.Tensor):
                        f_np = f.detach().cpu().numpy().flatten()
                    elif isinstance(f, np.ndarray):
                        f_np = f.flatten()
                    elif isinstance(f, list):
                        f_np = np.array(f).flatten()
                    feats.append(f_np)
             
        avg_feat = None
        if feats:
            # 确保所有特征维度一致
            try:
                avg_feat = np.mean(feats, axis=0)
            except Exception as e:
                print(f"特征均值计算失败: {e}")
                avg_feat = feats[0] # 兜底取第一个

        metadata[i] = {
            'nodes': [n['id'] for n in trk],
            'start_frame': start_node['frame'],
            'end_frame': end_node['frame'],
            'start_pos': start_node['pos'],
            'end_pos': end_node['pos'],
            'end_vel': end_node.get('velocity', np.array([0, 0])),
            'feat': avg_feat,
            'original_id': trk[0].get('raw_id')
        }

    prob = pulp.LpProblem("Refine_FastTracker", pulp.LpMaximize)
    edge_vars = {}
    
    # --- 超参数 ---
    time_window = config.get('time_window', 30) 
    dist_thresh = 150.0  
    w_dist = 20.0
    w_feat = 50.0
    min_score = 10.0 

    indices = list(metadata.keys())
    for i in indices:
        for j in indices:
            if i == j: continue
            m_i, m_j = metadata[i], metadata[j]
            gap = m_j['start_frame'] - m_i['end_frame']
            
            # 只连接时间顺序正确的片段
            if 1 <= gap <= time_window:
                # 运动预测：基于上一段结束位置和速度预测下一段起始位置
                pred_pos = m_i['end_pos'] + m_i['end_vel'] * gap
                dist = np.linalg.norm(pred_pos - m_j['start_pos'])
                
                if dist < dist_thresh:
                    # 运动得分：高斯核
                    score_pos = np.exp(-(dist**2) / (2 * (50**2))) * w_dist
                    
                    # 特征得分：余弦相似度
                    score_feat = 0
                    if m_i['feat'] is not None and m_j['feat'] is not None:
                        norm_i = np.linalg.norm(m_i['feat'])
                        norm_j = np.linalg.norm(m_j['feat'])
                        if norm_i > 1e-6 and norm_j > 1e-6:
                            sim = np.dot(m_i['feat'], m_j['feat']) / (norm_i * norm_j)
                            score_feat = max(0, sim) * w_feat
                    
                    total_score = score_pos + score_feat
                    
                    if total_score > min_score:
                        v = pulp.LpVariable(f"edge_{i}_{j}", 0, 1, pulp.LpBinary)
                        edge_vars[(i, j)] = (v, total_score)

    # 流量约束：每个节点最多一个入边和一个出边
    for k in indices:
        outs = [v for (i, j), (v, s) in edge_vars.items() if i == k]
        ins = [v for (i, j), (v, s) in edge_vars.items() if j == k]
        if outs: prob += pulp.lpSum(outs) <= 1
        if ins: prob += pulp.lpSum(ins) <= 1

    # 求解
    if edge_vars:
        prob += pulp.lpSum([v * s for (i, j), (v, s) in edge_vars.items()])
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        links = {i: j for (i, j), (v, s) in edge_vars.items() if pulp.value(v) > 0.5}
    else:
        links = {}

    # 轨迹重建
    final_tracks_nodes = []
    visited = set()
    
    # 1. 寻找链条起点并跟踪
    for i in indices:
        if i in visited: continue
        
        # 判断是否为链条起点
        is_start = True
        for head, tail in links.items():
            if tail == i:
                is_start = False
                break
        
        if is_start:
            curr_chain = []
            curr = i
            while curr is not None:
                curr_chain.extend(metadata[curr]['nodes'])
                visited.add(curr)
                curr = links.get(curr)
            if curr_chain:
                final_tracks_nodes.append(curr_chain)

    # 2. 兜底：处理孤立节点
    for i in indices:
        if i not in visited:
            final_tracks_nodes.append(metadata[i]['nodes'])

    return final_tracks_nodes

# def solve_ilp(graph, optim_config):
#     import pulp
#     import numpy as np

#     start_cost = optim_config.get('start_cost', 0.0)
#     end_cost = optim_config.get('end_cost', 0.0)
#     min_track_length = optim_config.get('min_track_length', None)
#     max_track_length = optim_config.get('max_track_length', None)

#     density_threshold = optim_config.get('density_threshold', 0.5)
#     min_distance_threshold = optim_config.get('min_distance_threshold', 3.0)

#     problem = pulp.LpProblem("TrackForming", pulp.LpMaximize)

#     edges = {}
#     positions = {}
#     nodes = list(graph.nodes())

#     for u, v in graph.edges():
#         var_name = f"edge_{u}_{v}"
#         edges[(u, v)] = pulp.LpVariable(var_name, cat=pulp.LpBinary)
#         positions[u] = graph.nodes[u]['pos']
#         positions[v] = graph.nodes[v]['pos']

#     if start_cost > 0 or end_cost > 0:
#         start_node = 'Start'
#         end_node = 'End'
#         graph.add_node(start_node)
#         graph.add_node(end_node)
#         for node in nodes:
#             var_start = f"edge_{start_node}_{node}"
#             var_end = f"edge_{node}_{end_node}"
#             edges[(start_node, node)] = pulp.LpVariable(var_start, cat=pulp.LpBinary)
#             edges[(node, end_node)] = pulp.LpVariable(var_end, cat=pulp.LpBinary)
#             graph.add_edge(start_node, node, weight=start_cost)
#             graph.add_edge(node, end_node, weight=end_cost)

#     problem += pulp.lpSum(edges[e] * graph.edges[e]['weight'] for e in edges)

#     # 入出度限制
#     for node in nodes:
#         incoming_edges = [edges[(u, node)] for u in graph.predecessors(node) if (u, node) in edges]
#         outgoing_edges = [edges[(node, v)] for v in graph.successors(node) if (node, v) in edges]
#         problem += pulp.lpSum(incoming_edges) <= 1, f"MaxIn_{node}"
#         problem += pulp.lpSum(outgoing_edges) <= 1, f"MaxOut_{node}"

#     if min_track_length is not None or max_track_length is not None:
#         track_vars = {}
#         track_id = 0

#         def add_track_length_constraints(current_node, current_length):
#             outgoing_edges = [edges[(current_node, v)] for v in graph.successors(current_node) if (current_node, v) in edges]
#             problem += track_var >= current_length, f"TrackLength_{current_node}"
#             for edge_var, successor_node in zip(outgoing_edges, graph.successors(current_node)):
#                 problem += edge_var <= track_var, f"EdgeTrack_{current_node}_{successor_node}"
#                 add_track_length_constraints(successor_node, current_length + 1)

#         for node in nodes:
#             incoming_edges = [edges[(u, node)] for u in graph.predecessors(node) if (u, node) in edges]
#             if len(incoming_edges) == 0:
#                 track_var = pulp.LpVariable(f"Track_{track_id}", lowBound=0, cat=pulp.LpInteger)
#                 track_id += 1
#                 add_track_length_constraints(node, 1)
#                 if min_track_length is not None:
#                     problem += track_var >= min_track_length, f"MinTrackLength_{node}"
#                 if max_track_length is not None:
#                     problem += track_var <= max_track_length, f"MaxTrackLength_{node}"

#     problem.solve()

#     if pulp.LpStatus[problem.status] == "Optimal":
#         # solution_edges: list of (u,v,pos_u,pos_v)
#         selected_edges = [
#             (u, v, positions[u], positions[v])
#             for u, v in edges
#             if edges[(u, v)].varValue > 0.5 and u in positions and v in positions
#             if u != 'Start' and v != 'End'
#         ]

#         # 构建轨迹(含edges)以进行密度检查
#         trajectories_with_edges = build_trajectories_with_edges(selected_edges)

#         filtered_edges = []
#         for traj_nodes, traj_positions, traj_edges in trajectories_with_edges:
#             density = compute_track_density(traj_positions)
#             total_displacement = np.linalg.norm(traj_positions[-1] - traj_positions[0])
#             if density <= density_threshold and total_displacement >= min_distance_threshold:
#                 # 保留该轨迹的所有edges
#                 filtered_edges.extend(traj_edges)

#         return filtered_edges
#     else:
#         return None
    
def build_trajectories_with_edges(selected_edges):
    """
    返回一个列表，每个元素为 (traj_nodes, traj_positions, traj_edges)
    traj_nodes: 轨迹节点序列 (u,v,...)
    traj_positions: 与traj_nodes对应的position序列的np.array(L, 2)
    traj_edges: 构成该轨迹的边列表[(u,v,pos_u,pos_v), ...]
    """
    from collections import defaultdict
    import numpy as np

    graph_dict = defaultdict(list)
    nodes_set = set()

    # 构建图结构
    for u, v, pos_u, pos_v in selected_edges:
        graph_dict[u].append((v, pos_u, pos_v))
        nodes_set.add(u)
        nodes_set.add(v)

    # 入度统计
    in_degree = {n:0 for n in nodes_set}
    for u in graph_dict:
        for (vv, pos_u, pos_v) in graph_dict[u]:
            in_degree[vv] += 1
    start_nodes = [n for n in nodes_set if in_degree[n] == 0]

    trajectories = []
    for start in start_nodes:
        # 重建轨迹:从start出发
        # 找到start点pos: 使用graph_dict[start]的第一条边的pos_u作为起始点
        traj_nodes = [start]
        traj_positions = []
        traj_edges = []

        if start in graph_dict and len(graph_dict[start]) > 0:
            # 使用第一条边的 pos_u 作为起始点位置
            first_edge = graph_dict[start][0]
            pos_start = first_edge[1]
            traj_positions.append(pos_start)
        else:
            # 如果start没有后继边，找到start的pos
            pos_start = None
            for (uu, vv, pu, pv) in selected_edges:
                if uu == start:
                    pos_start = pu
                    break
                if vv == start:
                    pos_start = pv
                    break
            if pos_start is None:
                # 无法找到start点pos,跳过
                continue
            traj_positions.append(pos_start)

        current = start
        while current in graph_dict and len(graph_dict[current]) == 1:
            next_node, pos_u, pos_v = graph_dict[current][0]
            traj_nodes.append(next_node)
            traj_positions.append(pos_v)
            traj_edges.append((current, next_node, pos_u, pos_v))
            current = next_node

        trajectories.append((traj_nodes, np.array(traj_positions), traj_edges))

    return trajectories


def compute_track_density(traj_positions):
    """
    计算轨迹密度
    traj_positions: numpy数组，形状为 (L, 2)
    """
    L = len(traj_positions)
    if L < 2:
        # 如果轨迹太短，没有后续点，就不视为异常
        return 0.0

    p_t = traj_positions[0]
    total_N = 0
    denominator = 0
    # k从1到L-1
    for k in range(1, L):
        p_tk = traj_positions[k]
        R_tk = np.linalg.norm(p_tk - p_t)
        # 在 {p_k, p_{k+1}, ..., p_{L-1}} 中统计位于半径 R_tk 内的点数 N_{t+k}
        sub_points = traj_positions[k:]  # k到末尾的点
        count_abnormal = np.sum(np.linalg.norm(sub_points - p_t, axis=1) <= R_tk)
        total_N += count_abnormal
        denominator += (L - k)  # (L-k)为剩余点数量

    if denominator == 0:
        return 0.0
    density = total_N / denominator
    return density


def apply_invalid_fragment_backtracking(traj_nodes, traj_positions, traj_edges, density_threshold, min_distance_threshold):
    """
    对存在异常的轨迹进行局部修正，剔除无效片段
    traj_nodes: 轨迹节点序列 (u,v,...)
    traj_positions: 与traj_nodes对应的position序列的np.array(L, 2)
    traj_edges: 构成该轨迹的边列表[(u,v,pos_u,pos_v), ...]
    返回修正后的有效轨迹边列表
    """

    L = len(traj_positions)
    if L < 2:
        return []

    # 逐步检查轨迹密度，从前到后
    valid_traj_edges = []
    current_traj_nodes = []
    current_traj_positions = []
    current_traj_edges = []

    for i in range(L):
        # 构建当前子轨迹
        current_traj_nodes.append(traj_nodes[i])
        current_traj_positions = traj_positions[:i+1]
        if i < L -1:
            edge = traj_edges[i]
            current_traj_edges.append(edge)

        if len(current_traj_positions) >= 2:
            density = compute_track_density(current_traj_positions)
            displacement = np.linalg.norm(current_traj_positions[-1] - current_traj_positions[0])
            if density > density_threshold or displacement < min_distance_threshold:
                # 异常点出现，进行回溯剔除
                # 保留当前轨迹到上一个有效点
                if i > 0:
                    # 保留前i个点
                    valid_traj_edges.extend(current_traj_edges[:-1])
                # 重置当前轨迹
                current_traj_nodes = [traj_nodes[i]]
                current_traj_positions = traj_positions[i:i+1]
                current_traj_edges = []
        else:
            # 轨迹长度不足2，不进行密度检查
            pass

    # 最后保留剩余的有效轨迹
    if len(current_traj_positions) >= 2:
        density = compute_track_density(current_traj_positions)
        displacement = np.linalg.norm(current_traj_positions[-1] - current_traj_positions[0])
        if density <= density_threshold and displacement >= min_distance_threshold:
            valid_traj_edges.extend(current_traj_edges)

    return valid_traj_edges

def parse_prediction(solution_edges):
    """ Parses the solution edges to create tracks with consistent IDs."""
    # 建立节点的前驱和后继映射
    node_successors = {}
    node_predecessors = {}
    nodes = set()
    node_centers = {}
    for u, v, pos_u, pos_v in solution_edges:
        node_successors[u] = v
        node_predecessors[v] = u
        nodes.add(u)
        nodes.add(v)
        node_centers[u] = pos_u
        node_centers[v] = pos_v

    # 查找起始节点（没有前驱的节点）
    start_nodes = [node for node in nodes if node not in node_predecessors]

    pred_tracks = {}
    frame_tracks = {}
    track_id_counter = 0
    visited_nodes = set()

    # 遍历每个起始节点，构建轨迹
    for start_node in start_nodes:
        node = start_node
        track_id = track_id_counter
        track_id_counter += 1
        while True:
            if node in visited_nodes:
                break
            visited_nodes.add(node)
            frame_id, cell_id = node
            center = node_centers.get(node)
            if center is None:
                break
            # 转换为左上角坐标
            tl_x = center[0] - center[2] / 2
            tl_y = center[1] - center[3] / 2
            # 更新 pred_tracks
            if track_id not in pred_tracks:
                pred_tracks[track_id] = {}
            pred_tracks[track_id][frame_id] = (tl_x, tl_y, center[2], center[3], False)
            # 更新 frame_tracks
            if frame_id not in frame_tracks:
                frame_tracks[frame_id] = []
            frame_tracks[frame_id].append((track_id, tl_x, tl_y, center[2], center[3], False))
            # 移动到下一个节点
            if node in node_successors:
                node = node_successors[node]
            else:
                break
    return pred_tracks, frame_tracks


def insert_virtual_nodes(pred_tracks_initial,
                         cells_by_t=None,  # unused, just keep signature
                         optim_config=None):
    """
    在“已有ILP轨迹碎片”基础上, 用 SORT 风格的逐帧KF重放 + 末帧虚拟预测 + IOU拼接.

    输入: 
      pred_tracks_initial: {tid: {frame: (x,y,w,h,is_v)}}
    输出:
      pred_tracks:         {tid: {frame: (x,y,w,h,is_v, vx, vy)}}
    """
    if optim_config is None:
        optim_config = {}
    max_age = optim_config.get('max_virtual', 10)
    iou_threshold = optim_config.get('iou_threshold', 0.3)

    # 1) 先对所有轨迹做 "replay_track_with_kf"
    #    => 让KF学到真实运动, 并输出 (vx,vy) 
    pred_tracks = {}
    kf_dict = {}  # 保存每条轨迹末尾的 KF (含最终速度)
    for tid, frames_map in pred_tracks_initial.items():
        kf_tracker, frames_map_out = replay_track_with_kf(frames_map)
        pred_tracks[tid] = frames_map_out
        kf_dict[tid] = kf_tracker  # 末帧对应的tracker

    # 2) 拼接: 对每条轨迹, 从其末帧开始, 做虚拟预测, 看是否能连接别的轨迹
    all_tids = sorted(pred_tracks.keys())
    merged_set = set()

    for tid in all_tids:
        if tid in merged_set:
            continue
        frames_sorted = sorted(pred_tracks[tid].keys())
        if not frames_sorted:
            continue

        # 取其最后一帧
        end_frame = frames_sorted[-1]
        # 这个KF已经在 replay过程中 更新到末帧的状态
        kf_tracker = kf_dict[tid]
        time_since_update = 0
        cur_frame = end_frame

        # 如果 ILP 已经把这轨迹和下一帧关联, 不需要虚拟预测
        # 但如何判断? 
        #   => 看 pred_tracks[tid] 是否还有 frame = end_frame+1
        #   如果有的话, 说明没断. 
        # 这里简单: 如果 (end_frame+1) 不在 frames_sorted, 说明断了, 要虚拟预测
        while (cur_frame+1) not in frames_sorted and time_since_update <= max_age:
            # step to next frame
            cur_frame += 1
            time_since_update += 1

            # predict
            pred_box = kf_tracker.predict()  # shape=(4,)
            
            # 在 cur_frame 中找 "起始帧==cur_frame" 的其它碎片
            candidate_tids = []
            for other_tid in all_tids:
                if other_tid == tid or other_tid in merged_set:
                    continue
                frames2 = sorted(pred_tracks[other_tid].keys())
                if not frames2:
                    continue
                if frames2[0] == cur_frame:
                    candidate_tids.append(other_tid)

            if len(candidate_tids)==0:
                # 没有碎片在这帧开始 => 继续
                if time_since_update>max_age:
                    break
                continue
            
            # 计算 IOU 
            best_iou = 0.
            best_tid2 = None
            for t2 in candidate_tids:
                frames2 = sorted(pred_tracks[t2].keys())
                stf = frames2[0]
                (xx, yy, ww, hh, isv2, vx2, vy2) = pred_tracks[t2][stf]
                box2 = box_from_xywh(xx, yy, ww, hh)
                iou_val = iou_1v1(pred_box, box2)
                if iou_val>best_iou:
                    best_iou = iou_val
                    best_tid2 = t2
            
            if best_iou >= iou_threshold and best_tid2 is not None:
                # update with that box
                kf_tracker.update(np.array(box2, dtype=np.float32))

                # 合并 best_tid2 整段
                for f2 in sorted(pred_tracks[best_tid2].keys()):
                    pred_tracks[tid][f2] = pred_tracks[best_tid2][f2]
                merged_set.add(best_tid2)
                # end
                break
            else:
                # no match => next
                if time_since_update>max_age:
                    break

    return pred_tracks


def post_process_tracks(pred_tracks, optim_config):
    """
    1) 过滤真实节点数 < min_track_length 的轨迹
    2) 保留虚拟节点，以供可视化或评估
    假设 pred_tracks[tid][frame] 可能是:
      (x, y, w, h, is_virtual) 或 (x, y, w, h, is_virtual, vx, vy)
    """
    min_track_length = optim_config.get('min_track_length', 2)
    final_tracks = {}
    
    for tid, frames_dict in pred_tracks.items():
        real_count = 0
        for fr, box in frames_dict.items():
            # 根据长度判断:
            if len(box) >= 5:
                # box[:5] => (x, y, w, h, isv)
                # isv = box[4]
                # 如果是 bool => True/False
                isv = box[4]
                # 若它是 bool => 说明 box[4] 为 is_virtual
                # 真实节点 => if not isv
                if isinstance(isv, bool):
                    if not isv:
                        real_count += 1
                else:
                    # box[4] 不是 bool => 说明这条数据没有 is_virtual
                    # => 视为真实节点
                    real_count += 1
            else:
                # 若 len(box)<5 => 可能是 (x,y,w,h) => 视为真实节点
                real_count += 1
        
        if real_count >= min_track_length:
            final_tracks[tid] = frames_dict
    
    return final_tracks

def solve_ilp_initial(G):
    """
    初始ILP求解，仅基于真实节点的图
    """
    problem = pulp.LpProblem("InitialTrackForming", pulp.LpMaximize)
    edges_var = {}
    for (u, v) in G.edges():
        var_name = f"edge_{u}_{v}"
        edges_var[(u, v)] = pulp.LpVariable(var_name, cat=pulp.LpBinary)

    # 目标函数: 最大化所有边的权重之和
    problem += pulp.lpSum([edges_var[e] * G.edges[e]['weight'] for e in edges_var])

    # 入度和出度约束: 每个节点的入度和出度均不超过1
    for node in G.nodes():
        incoming = [edges_var[(u, node)] for u in G.predecessors(node) if (u, node) in edges_var]
        outgoing = [edges_var[(node, v)] for v in G.successors(node) if (node, v) in edges_var]
        problem += pulp.lpSum(incoming) <= 1, f"MaxIn_{node}"
        problem += pulp.lpSum(outgoing) <= 1, f"MaxOut_{node}"

    # 求解ILP
    problem.solve()

    if pulp.LpStatus[problem.status] != "Optimal":
        return None

    # 提取选中的边
    selected_edges = []
    for (u, v), var in edges_var.items():
        if var.varValue > 0.5:
            pos_u = G.nodes[u]['pos']
            pos_v = G.nodes[v]['pos']
            selected_edges.append((u, v, pos_u, pos_v))
    return selected_edges

def solve_ilp_final(G, optim_config):
    """
    最终ILP求解，包含虚拟节点的图
    """
    density_threshold = optim_config.get('density_threshold', 0.45)
    min_distance_threshold = optim_config.get('min_distance_threshold', 2)
    problem = pulp.LpProblem("FinalTrackForming", pulp.LpMaximize)
    edges_var = {}
    for (u, v) in G.edges():
        var_name = f"edge_{u}_{v}"
        edges_var[(u, v)] = pulp.LpVariable(var_name, cat=pulp.LpBinary)

    # 目标函数: 最大化所有边的权重之和
    problem += pulp.lpSum([edges_var[e] * G.edges[e]['weight'] for e in edges_var])

    # 入度和出度约束: 每个节点的入度和出度均不超过1
    for node in G.nodes():
        incoming = [edges_var[(u, node)] for u in G.predecessors(node) if (u, node) in edges_var]
        outgoing = [edges_var[(node, v)] for v in G.successors(node) if (node, v) in edges_var]
        problem += pulp.lpSum(incoming) <= 1, f"MaxIn_{node}"
        problem += pulp.lpSum(outgoing) <= 1, f"MaxOut_{node}"

    # 求解ILP
    problem.solve()

    if pulp.LpStatus[problem.status] != "Optimal":
        return None

    # 提取选中的边
    selected_edges = []
    for (u, v), var in edges_var.items():
        if var.varValue > 0.5:
            pos_u = G.nodes[u]['pos']
            pos_v = G.nodes[v]['pos']
            selected_edges.append((u, v, pos_u, pos_v))

    # 构建轨迹(含edges)以进行密度检查
    trajectories_with_edges = build_trajectories_with_edges(selected_edges)

    filtered_edges = []
    for traj_nodes, traj_positions, traj_edges in trajectories_with_edges:
        # 轨迹密度计算
        density = compute_track_density(traj_positions)
        # 轨迹位移计算
        total_displacement = np.linalg.norm(traj_positions[-1] - traj_positions[0])
        # 轨迹密度与位移检查
        if density <= density_threshold and total_displacement >= min_distance_threshold:
            # 保留该轨迹的所有edges
            filtered_edges.extend(traj_edges)

    return selected_edges


def rebuild_frame_tracks_from_pred_tracks(pred_tracks):
    """
    pred_tracks: {tid: {frame: (x,y,w,h,is_v[, vx,vy,...])}}
    => 返回 frame_tracks: {frame: [(tid, x,y,w,h,is_v[, vx,vy,...])]}

    如果后续写文件只需 tid, x,y,w,h,isv,可将 vx,vy 等保留在 track tuple 里看需求。
    """
    frame_tracks = {}
    for tid, frames_dict in pred_tracks.items():
        for fr, box in frames_dict.items():
            # box 至少5个维度: (x,y,w,h,is_v), 可能≥7 (vx,vy)
            if len(box) < 5:
                # 如果不足5 => 视为 (x,y,w,h)? => 补 is_v=False
                x, y, w, h = box[:4]
                is_v = False
                # vx, vy = 0,0  (可选)
                new_track_tuple = (tid, x, y, w, h, is_v)
            else:
                # 取前5个
                x, y, w, h, is_v = box[:5]
                # 如果还存在 vx, vy => box[5], box[6]
                vx, vy = 0.0, 0.0
                if len(box) >= 7:
                    vx, vy = box[5], box[6]
                # 可以选择保留
                # new_track_tuple = (tid, x, y, w, h, is_v, vx, vy)
                # 如果写文件只要6个 => (tid,x,y,w,h,is_v)
                new_track_tuple = (tid, x, y, w, h, is_v)
            
            if fr not in frame_tracks:
                frame_tracks[fr] = []
            frame_tracks[fr].append(new_track_tuple)
    
    return frame_tracks

def convert_frame_tracks_to_mot_eval(frame_tracks):
    """
    将按帧组织的轨迹数据转换为 MOT Challenge 评估所需的 NumPy 数组格式。

    输入: 
        frame_tracks: {frame: [(tid, x, y, w, h, is_v)]}
            - x, y, w, h 是 ILP 使用的中心点 + 宽度/高度格式（需要确认）
            - tid, frame 必须是整数

    输出: 
        mot_data_array: NumPy 数组, shape=(N, 10)
        格式: [frame, id, x_left, y_top, w, h, score, class, visibility, -1]
    """
    mot_lines = []
    
    # 确保按帧号顺序处理
    sorted_frames = sorted(frame_tracks.keys())
    
    # 假设：
    # 1. frame_tracks 中的 x, y 是边界框中心点坐标。
    # 2. 您需要为 ILP 插入的 'is_v=True' 虚拟节点设置 score=0 或忽略。
    # 3. 如果原始输入没有提供 score，我们默认为 1.0。
    
    for frame_id in sorted_frames:
        tracks_in_frame = frame_tracks[frame_id]
        
        for track_tuple in tracks_in_frame:
            # track_tuple: (tid, x_center, y_center, w, h, is_v, [score/conf])
            
            # 提取基础数据
            tid = track_tuple[0]
            x_center = track_tuple[1]
            y_center = track_tuple[2]
            w = track_tuple[3]
            h = track_tuple[4]
            is_v = track_tuple[5]
            
            # 尝试获取置信度/分数 (如果您的 rebuild_frame_tracks_from_pred_tracks 保留了它)
            # 假设 score/conf 是元组的第7个元素 (index 6)，如果存在的话
            score = track_tuple[6] if len(track_tuple) > 6 else 1.0
            
            # --- 核心转换：中心点(x_center, y_center) 到 左上角(x_left, y_top) ---
            x_left = x_center - w / 2.0
            y_top = y_center - h / 2.0
            
            # --- MOT 格式填充 ---
            
            # 1. 跳过虚拟节点 (可选, 推荐跳过，除非评估工具要求保留)
            if is_v:
                # 虚拟节点通常不应该用于 MOT 评估，可以跳过或设置低分
                # 如果要保留，可以设置 score=0.0，但跳过更常见
                continue 
            
            # 2. 构造 MOT 行数据
            # [frame, id, x_left, y_top, w, h, score, class, visibility, -1]
            mot_line = [
                frame_id, 
                tid, 
                x_left, 
                y_top, 
                w, 
                h, 
                score, 
                1,      # class_id: 1 (行人/默认)
                -1,     # visibility: -1 (未使用)
                -1      # 保留字段
            ]
            
            mot_lines.append(mot_line)

    # 转换为 NumPy 数组
    return np.array(mot_lines, dtype=np.float32)

def iou_1v1(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0., xB - xA)
    interH = max(0., yB - yA)
    interArea = interW * interH
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = areaA + areaB - interArea
    return interArea / union if union>0 else 0.0

def box_from_xywh(x, y, w, h):
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1,y1,x2,y2]

def box_iou(boxA, boxB):
    """
    计算两个边界框之间的 IoU (Intersection over Union)。
    边界框格式: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# --- 辅助函数：转换 Cell 位置到 [x1, y1, x2, y2] ---
def cell_pos_to_box(position):
    """
    将 Cell 的位置格式 [xc, yc, w, h] 转换为 [x1, y1, x2, y2]
    """
    xc, yc, w, h = position
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return np.array([x1, y1, x2, y2], dtype=np.float32)

# --- 核心函数 ---
# def convert_tracklets_to_internal_edges(initial_tracklets, cells_by_t, optim_config, iou_thresh=0.8):
#     """
#     将 FastTracker 生成的扁平化检测列表转换为 ILP 图中的高权重边。
    
#     Args:
#         initial_tracklets: FastTracker.get_tracklets() 的输出。
#                          结构为 [(frame_id, tid, x1, y1, x2, y2, ...), ...] (元组列表)
#         cells_by_t: 由 build_videos_cells_by_t 生成的 ILP 节点字典。
#         optim_config: 优化配置字典，需要包含 'tracklet_weight' (高权重奖励值)。
#         iou_thresh: 用于匹配 FastTracker 的 Box 和 Cell Box 的 IoU 阈值。

#     Returns:
#         tracklet_edges: 结构如 [(u_cell_tid, v_cell_tid, cost), ...]
#     """
#     tracklet_edges = []
    
#     # 1. 重构 FastTracker 数据结构
#     # 由于 initial_tracklets 是扁平列表，我们需要按 (frame_id, tid) 建立索引
#     fast_track_index = defaultdict(lambda: defaultdict(dict)) 
#     # 结构: fast_track_index[frame_id][tid] = [x1, y1, x2, y2]
    
#     for item in initial_tracklets:
#         # 验证 tracklet item 是元组且长度足够 (至少包含 frame_id, tid, x1, y1, x2, y2)
#         if not isinstance(item, tuple) or len(item) < 6:
#             continue
            
#         # 使用索引访问：frame_id [0], tid [1], x1 [2], y1 [3], x2 [4], y2 [5]
#         frame_id, tid, x1, y1, x2, y2 = item[0], item[1], item[2], item[3], item[4], item[5]
        
#         box = np.array([x1, y1, x2, y2])
#         fast_track_index[frame_id][tid] = box

#     # 2. 构建 ILP Cells 的反向索引： frame_id -> [cell_tid, cell_box(x1y1x2y2), ...]
#     # 目的：快速查找给定帧和边界框的对应 Cell ID
#     cells_index = defaultdict(list)
#     for frame_id, cell_list in cells_by_t.items():
#         for cell_tuple in cell_list:
#             # cell_tuple: (tid, position[xc, yc, w, h], velocity, confidence, appearance)
#             tid, position, _, _, _ = cell_tuple
#             box = cell_pos_to_box(position)
#             cells_index[frame_id].append({'tid': tid, 'box': box})

#     # 3. 遍历 FastTracker 的轨迹索引，生成高权重边
#     TRACKLET_REWARD = optim_config.get('tracklet_weight', 100.0)
#     TRACKLET_COST = TRACKLET_REWARD # ILP 成本是负奖励

#     sorted_frames = sorted(fast_track_index.keys())

#     for i in range(len(sorted_frames) - 1):
#         frame_i = sorted_frames[i]
#         frame_j = sorted_frames[i+1]
        
#         # 我们只考虑连续帧的关联 (FastTracker 的边)
#         if frame_j != frame_i + 1:
#             continue
            
#         tids_i = fast_track_index[frame_i].keys()
        
#         # 遍历在 frame_i 中存在的所有轨迹 ID
#         for tid in tids_i:
#             # 检查这个 ID 在下一帧 frame_j 是否也存在
#             if tid in fast_track_index[frame_j]:
#                 # 这是一个由 FastTracker 确定的连续关联
#                 obs_i_box = fast_track_index[frame_i][tid]
#                 obs_j_box = fast_track_index[frame_j][tid]

#                 # --- A. 匹配 obs_i 到 Cell_i (frame_i) ---
#                 cells_i = cells_index.get(frame_i, [])
#                 best_iou_i = -1
#                 best_cell_tid_i = None
                
#                 for cell in cells_i:
#                     iou = box_iou(obs_i_box, cell['box'])
#                     if iou > best_iou_i:
#                         best_iou_i = iou
#                         best_cell_tid_i = cell['tid']
                
#                 # --- B. 匹配 obs_j 到 Cell_j (frame_j) ---
#                 cells_j = cells_index.get(frame_j, [])
#                 best_iou_j = -1
#                 best_cell_tid_j = None
                
#                 for cell in cells_j:
#                     iou = box_iou(obs_j_box, cell['box'])
#                     if iou > best_iou_j:
#                         best_iou_j = iou
#                         best_cell_tid_j = cell['tid']

#                 # --- C. 确定边 ---
#                 if best_cell_tid_i is not None and best_cell_tid_j is not None and \
#                    best_iou_i >= iou_thresh and best_iou_j >= iou_thresh:
                    
#                     # 如果 FastTracker 关联的两点都成功匹配到 ILP Cell，则添加一条高权重边
#                     # ILP 边格式: (u_cell_tid, v_cell_tid, cost)
#                     tracklet_edges.append((best_cell_tid_i, best_cell_tid_j, TRACKLET_COST))

#     return tracklet_edges

def extract_features_for_class_1(ret, feature_map):
    h_map, w_map = feature_map.shape[2:]  # 特征图的高宽
    f1 = []  # 存储类别号为 1 的外观特征

    if 1 in ret:  # 检查类别号 1 是否存在
        boxes = ret[1]
        for box in boxes:
            x_min, y_min, x_max, y_max, conf = box

            # 检测框合法性检查
            if x_max <= x_min or y_max <= y_min:
                f1.append(torch.zeros(feature_map.size(1)))  # 填充零特征
                continue

            # 将检测框坐标映射到特征图坐标
            x_min = int(max(0, x_min * w_map / feature_map.size(-1)))
            x_max = int(min(w_map, x_max * w_map / feature_map.size(-1)))
            y_min = int(max(0, y_min * h_map / feature_map.size(-2)))
            y_max = int(min(h_map, y_max * h_map / feature_map.size(-2)))

            # 检查裁剪区域有效性
            if x_max <= x_min or y_max <= y_min:
                f1.append(torch.zeros(feature_map.size(1)))  # 填充零特征
                continue

            # 裁剪特征图
            cropped_feature = feature_map[:, :, y_min:y_max, x_min:x_max]

            # 特殊处理空裁剪区域
            if cropped_feature.numel() == 0:
                feature_vector = torch.zeros(feature_map.size(1))  # 填充零特征
            else:
                pooled_feature = F.adaptive_avg_pool2d(cropped_feature, (1, 1))
                feature_vector = pooled_feature.view(-1)

            # 添加特征向量
            f1.append(feature_vector)

    # 将类别号 1 的外观特征加入 ret
    ret['f1'] = f1
    return ret

def read_flow_flo(filename):
    """
    Read Middlebury .flo file.
    Format: 4 bytes tag 'PIEH', width (int32), height (int32), data (float32)
    """
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            print(f"Magic number incorrect. Invalid .flo file: {filename}")
            return None
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape to (H, W, 2) then transpose to (2, H, W) to match your logic
        flow = np.resize(data, (h, w, 2)).transpose(2, 0, 1)
        return flow

def build_videos_cells_by_t(
    res,
    video_gt_dict,
    first_appear_dict,
    area_percentage=10
):
    """
    res: {image_path -> {1: array_of_detection, 'f1': features}}
    video_gt_dict: 同 parse_all_ground_truths 返回
    first_appear_dict: { (video_name, obj_id): first_frame_id }
    area_percentage: 用于光流区域 => 10表示10%
    
    返回:
      videos_cells_by_t = {
        video_name: {
          frame_number: [
             (tid, position(4d), velocity(2d), conf, appearance),
             ...
          ]
        }
      }
    """
    videos_cells_by_t = {}
    tid_counters = {}
    
    # 定义光流数据集根目录 (根据你的描述)
    FLOW_ROOT = "/home/liangcx/datasets/crop_datasets/flow"

    for image_path, detections_dict in res.items():
        # 1) 提取 video_name & frame_number
        path_parts = image_path.split('/')
        try:
            test_index = path_parts.index('test')
            video_name = path_parts[test_index + 1]
        except ValueError:
            print(f"无法在路径中找到 'test': {image_path}")
            continue
        
        if video_name not in videos_cells_by_t:
            videos_cells_by_t[video_name] = {}
            tid_counters[video_name] = 0
        cells_by_t = videos_cells_by_t[video_name]
        tid_counter = tid_counters[video_name]
        
        image_name = os.path.basename(image_path)
        frame_number_str = os.path.splitext(image_name)[0]
        frame_number = int(frame_number_str)  # 这里假设 => 1-based
        
        # 2) 构建 detection_list
        detection_list = []
        
        boxes = detections_dict[1]       # shape [N,5] => (x_min,y_min,x_max,y_max,conf)
        features = detections_dict['f1'] # shape [N, ...]
        
        for i, (box, feat) in enumerate(zip(boxes, features)):
            x_min, y_min, x_max, y_max, confidence = box
            if confidence<0.3:
                continue
            x_center = 0.5*(x_min+x_max)
            y_center = 0.5*(y_min+y_max)
            w = x_max - x_min
            h = y_max - y_min
            
            # => 先 placeholder flow_x,flow_y=0
            # 后面计算光流(见下)
            flow_x, flow_y = 0.0, 0.0
            
            # we will fill flow_x,flow_y after area flow
            # build detection => (tid, position, velocity, confidence, feat)
            # tid:
            tid = tid_counter
            tid_counter += 1
            
            position = np.array([x_center,y_center,w,h], dtype=np.float32)
            velocity = np.array([flow_x,flow_y], dtype=np.float32)
            detection_list.append((tid, position, velocity, confidence, feat))
        
        # 3) 计算光流 => for each detection => 取 [region_x_min,region_x_max,...] 并 np.mean
        #    这里和您原先代码一样
        flow_path = os.path.join(FLOW_ROOT, video_name, f"{frame_number_str}.flo")
        if os.path.exists(flow_path):
            # flow_map = np.load(flow_path)  # shape [2, H, W]
            flow_map = read_flow_flo(flow_path) # Result shape: [2, H, W]
            flow_channels, flow_h, flow_w = flow_map.shape
            
            # 逐个更新 detection 的 flow
            new_detection_list = []
            for (tid, pos, vel, conf, feat) in detection_list:
                x_c, y_c, ww, hh = pos
                # area
                region_w = ww * np.sqrt(area_percentage/100.0)
                region_h = hh * np.sqrt(area_percentage/100.0)
                
                # region_x_min ...
                region_x_min = max(int(x_c - region_w/2), int(x_c - ww/2))
                region_y_min = max(int(y_c - region_h/2), int(y_c - hh/2))
                region_x_max = min(int(x_c + region_w/2), int(x_c + ww/2))
                region_y_max = min(int(y_c + region_h/2), int(y_c + hh/2))
                
                # clip to flow range
                region_x_min = np.clip(region_x_min, 0, flow_w-1)
                region_y_min = np.clip(region_y_min, 0, flow_h-1)
                region_x_max = np.clip(region_x_max, 0, flow_w-1)
                region_y_max = np.clip(region_y_max, 0, flow_h-1)
                
                if region_x_max>=region_x_min and region_y_max>=region_y_min:
                    flow_region = flow_map[:, region_y_min:region_y_max+1, region_x_min:region_x_max+1]
                    fx = np.mean(flow_region[0])
                    fy = np.mean(flow_region[1])
                else:
                    fx, fy = 0.0,0.0
                
                velocity = np.array([fx,fy], dtype=np.float32)
                new_detection_list.append( (tid, pos, velocity, conf, feat) )
                # print(f"Video: {video_name}, Frame: {frame_number}, Detection: (tid={tid}, pos={pos}, vel={velocity}, conf={conf})")
            detection_list = new_detection_list
        else:
            # 不存在 => 保持velocity=0
            print(f"Setting velocity to zero for all detections in this frame.")
            pass
        
        # 4) 若 frame_number==1 => 用 GT 替换 detection_list
        #    (请注意frame_number可能是1-based,如果您想0-based,请改 if frame_number==0)
        if frame_number==1:
            detection_list.clear()
            # 读取 gt => video_gt_dict[video_name][1]
            # 可能为空
            if frame_number in video_gt_dict.get(video_name, {}):
                gtlist = video_gt_dict[video_name][frame_number]
                for (obj_id, gxmin, gymin, gw, gh, gconf, gvx, gvy) in gtlist:
                    tid = tid_counter
                    tid_counter+=1
                    x_center = gxmin + 0.5*gw
                    y_center = gymin + 0.5*gh
                    position = np.array([x_center,y_center,gw,gh], dtype=np.float32)
                    velocity = np.array([gvx,gvy], dtype=np.float32)
                    
                    # appearance先写None
                    detection_list.append( (tid, position, velocity, gconf, None) )
        
        # 5) 处理 "首次出现" => 用 GT => 删除相交>10%
        #    先看 gt 里有没有 frame_number
        if frame_number in video_gt_dict.get(video_name, {}):
            gtlist = video_gt_dict[video_name][frame_number]
            filtered_list = detection_list[:]  # 先copy
            for (obj_id, gxmin, gymin, gw, gh, gconf, gvx, gvy) in gtlist:
                # 看 first_appear_dict
                if (video_name, obj_id) in first_appear_dict:
                    first_appear_frame = first_appear_dict[(video_name, obj_id)]
                    if first_appear_frame == frame_number:
                        # => 先删除相交>10%
                        new_filtered = []
                        gt_box = (gxmin, gymin, gw, gh)
                        for (dtid, dpos, dvel, dconf, dfeat) in filtered_list:
                            dxmin = dpos[0] - dpos[2]/2
                            dymin = dpos[1] - dpos[3]/2
                            dw    = dpos[2]
                            dh    = dpos[3]
                            det_box = (dxmin, dymin, dw, dh)
                            if overlap_exceed_10pct(det_box, gt_box):
                                # skip
                                continue
                            else:
                                new_filtered.append( (dtid, dpos, dvel, dconf, dfeat) )
                        
                        filtered_list = new_filtered
                        
                        # 再添加这个 GT
                        tid = tid_counter
                        tid_counter+=1
                        x_center = gxmin + 0.5*gw
                        y_center = gymin + 0.5*gh
                        position = np.array([x_center,y_center,gw,gh], dtype=np.float32)
                        velocity = np.array([gvx,gvy], dtype=np.float32)
                        filtered_list.append( (tid, position, velocity, gconf, None) )
            
            detection_list = filtered_list
        
        # 6) 将 detection_list 放入 cells_by_t
        if frame_number not in cells_by_t:
            cells_by_t[frame_number] = []
        cells_by_t[frame_number].extend(detection_list)
        
        # 更新 tid
        tid_counters[video_name] = tid_counter
    
    return videos_cells_by_t


def global_tracker(cells_by_t_input, optim_configs):
    """
    全局跟踪主函数
    tracks: {frame: [(tid, x,y,w,h,is_v)]}
    optim_config: 优化配置字典
    返回:
      pred_tracks: {tid: {frame: (x,y,w,h,is_v,vx,vy)}}
      frame_tracks: {frame: [(tid, x,y,w,h,is_v)]}
    """
    # 1) 构建 cells_by_t
    cells_by_t = cells_by_t_input


    # 2) 创建图
    graph = create_graph(cells_by_t, optim_configs)

    # 3) 求解 ILP
    solution_edges = solve_ilp(graph, optim_configs)
    if solution_edges is None:
        return {}, {}

    # 4) 解析预测结果
    pred_tracks_initial, frame_tracks_initial = parse_prediction(solution_edges)

    # 5) 插入虚拟节点
    pred_tracks_with_virtuals = insert_virtual_nodes(pred_tracks_initial,
                                                     cells_by_t,
                                                     optim_configs)

    # 6) 后处理轨迹
    pred_tracks_final = post_process_tracks(pred_tracks_with_virtuals,
                                            optim_configs)

    # 7) 重建 frame_tracks
    frame_tracks_final = rebuild_frame_tracks_from_pred_tracks(pred_tracks_final)

    return pred_tracks_final, frame_tracks_final
