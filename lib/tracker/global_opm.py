import networkx as nx
import numpy as np
import cv2
from scipy.spatial import KDTree
import networkx as nx
import numpy as np
import scipy.linalg
from sklearn.metrics.pairwise import cosine_similarity
import pulp
from collections import defaultdict
import os
import torch.nn.functional as F
import torch
import re
import matplotlib.pyplot as plt
from .kalman_filter import KalmanFilter
import math
from .matching import linear_assignment

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

# ==========================================
# 自定义卡尔曼滤波器 (显式更名以防冲突)
# ==========================================
class MyKalmanFilter(object):
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [self._std_weight_position * mean[3]] * 2 + [1e-2, self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3]] * 2 + [1e-5, self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [self._std_weight_position * mean[3]] * 2 + [1e-1, self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T, check_finite=False).T
        innovation = measurement - projected_mean
        return mean + np.dot(innovation, kalman_gain.T), covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))

    def gating_distance(self, mean, covariance, measurements):
        """计算马氏距离"""
        mean, covariance = self.project(mean, covariance)
        d = measurements - mean
        cholesky_factor = np.linalg.cholesky(covariance)
        z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
        return np.sum(z * z, axis=0)

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

        self.kf.x[:4] = convert_bbox_to_z(bbox)
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

import networkx as nx
import numpy as np
import math

def compute_iou(box1, box2):
    """计算两个边界框(tlwh格式)的 IoU"""
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    iou = intersection_area / float(area1 + area2 - intersection_area)
    return iou

def tlwh_to_xyah(tlwh):
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret

def compute_iou_x1y1x2y2(boxA, boxB):
    # 计算两个 [x1, y1, x2, y2] 框的 IoU
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def build_global_graph_advanced(all_video_tracks, raw_detections, appearance_map):
    G = nx.DiGraph()
    
    # --- 严格参数调优 ---
    w_app = 25.0          # 再次提高 ReID 权重，只有很像才连
    max_time_gap = 40     # 允许更长的遮挡修复
    search_radius = 120   # 缩小搜索半径，减少误连的可能性
    min_det_score = 0.65  # 显著提高门槛！只允许高质量检测进入，消灭 FP
    
    track_history = {} 
    nodes_by_frame = {}

    # 1. 录入原始轨迹（基石权重设为极大）
    for item in all_video_tracks:
        f_id, t_id, x, y, w, h, score = int(item[0]), int(item[1]), *map(float, item[2:7])
        bbox = np.array([x, y, w, h])
        feat = appearance_map.get((f_id, t_id))
        
        node_id = f"T_{t_id}_{f_id}"
        G.add_node(node_id, track_id=t_id, frame_id=f_id, pos=bbox, appearance=feat, node_type='track', score=score)
        nodes_by_frame.setdefault(f_id, []).append(node_id)
        
        if t_id not in track_history:
            track_history[t_id] = {'frames': [], 'nodes': [], 'boxes': [], 'feats': []}
        track_history[t_id]['frames'].append(f_id)
        track_history[t_id]['nodes'].append(node_id)
        track_history[t_id]['boxes'].append(bbox)
        track_history[t_id]['feats'].append(feat)

    # 2. 内部边（绝对保护：权重 500）
    for t_id, data in track_history.items():
        indices = np.argsort(data['frames'])
        for i in range(len(indices) - 1):
            u, v = data['nodes'][indices[i]], data['nodes'][indices[i+1]]
            # 权重从 500 降到 20-30。
            # 这样如果这两帧之间 ReID 突变或距离突变，求解器就有权断开它们。
            G.add_edge(u, v, weight=25.0)
            
    # 3. 提取状态
    track_ends = {t_id: data['nodes'][np.argmax(data['frames'])] for t_id, data in track_history.items()}
    track_starts = {t_id: data['nodes'][np.argmin(data['frames'])] for t_id, data in track_history.items()}

    # 4. 检测框激活（加入“孤立检测惩罚”逻辑）
    for f_id, dets in raw_detections.items():
        for idx, det in enumerate(dets):
            d_bbox, d_score = np.array(det[:4]), det[4]
            if d_score < min_det_score: continue # 只有高分检测才配进图
            
            # 只有在现有轨迹“可能消失”的时空范围内，才激活该检测
            is_valid_candidate = False
            for t_id, end_id in track_ends.items():
                u_node = G.nodes[end_id]
                dt = f_id - u_node['frame_id']
                if 0 < dt <= max_time_gap:
                    dist = np.linalg.norm(u_node['pos'][:2] - d_bbox[:2])
                    if dist < search_radius:
                        is_valid_candidate = True; break
            
            if is_valid_candidate:
                node_id = f"D_{idx}_{f_id}"
                feat = appearance_map.get(f"det_{idx}_{f_id}")
                # 给检测节点一个负的基础权重（惩罚项），除非有强的边连进来，否则 ILP 不会选它
                G.add_node(node_id, track_id=-1, frame_id=f_id, pos=d_bbox, 
                           appearance=feat, score=d_score, node_type='det')
                nodes_by_frame.setdefault(f_id, []).append(node_id)

    # 5. 关联边逻辑 (重点：轨迹合并权重 > 检测桥接)
    for t_id_u, end_id in track_ends.items():
        u_data = G.nodes[end_id]
        for f_v in range(u_data['frame_id'] + 1, u_data['frame_id'] + max_time_gap + 1):
            if f_v not in nodes_by_frame: continue
            for v_id in nodes_by_frame[f_v]:
                v_data = G.nodes[v_id]
                if v_data.get('track_id') == t_id_u: continue
                
                dist = np.linalg.norm(u_data['pos'][:2] - v_data['pos'][:2])
                if dist > search_radius: continue
                
                sim = 0
                if u_data['appearance'] is not None and v_data['appearance'] is not None:
                    sim = cosine_similarity([u_data['appearance']], [v_data['appearance']])[0][0]
                
                if sim > 0.7: # 极高 ReID 要求
                    weight = w_app * (sim**2)
                    # 如果是合并两个已有轨迹，大幅奖励
                    if v_data.get('node_type') == 'track':
                        weight += 50.0 
                    G.add_edge(end_id, v_id, weight=weight)
    return G

def solve_ilp_fa(graph, optim_config=None):
    import pulp
    import numpy as np

    problem = pulp.LpProblem("Denoise_and_Link", pulp.LpMaximize)
    
    # --- 核心参数调整 ---
    # 每一帧的“生存成本”。如果一个框的得分或连接收益抵不上这 10 分，就会被删除
    node_survival_cost = -15.0  
    
    edges = {}
    nodes = [n for n in graph.nodes() if n not in ['Start', 'End']]
    
    # 1. 变量定义
    for u, v in graph.edges():
        edges[(u, v)] = pulp.LpVariable(f"e_{u}_{v}", cat=pulp.LpBinary)

    # 2. 虚拟节点逻辑：这里是去噪的关键
    start_node, end_node = 'Start', 'End'
    for node in nodes:
        edges[(start_node, node)] = pulp.LpVariable(f"s_{node}", cat=pulp.LpBinary)
        edges[(node, end_node)] = pulp.LpVariable(f"e_{node}", cat=pulp.LpBinary)
        
        # 节点基础收益 = 节点自身置信度得分 + 生存成本(负值)
        # 如果 score=0.4, cost=-15, 那么这个点本身就亏了 14.6 分，除非它能连接到别的点来弥补
        node_score = graph.nodes[node].get('score', 0.5) * 10.0
        s_w = node_score + node_survival_cost
        
        # 如果是纯检测框，加大惩罚，防止其独立成轨迹
        if graph.nodes[node].get('node_type') == 'det':
            s_w -= 30.0 
            
        graph.add_edge(start_node, node, weight=s_w)
        graph.add_edge(node, end_node, weight=0.0)

    # 3. 目标函数
    problem += pulp.lpSum(edges[e] * graph.edges[e].get('weight', 0) for e in edges)

    # 4. 约束
    for node in nodes:
        in_es = [edges[(u, node)] for u in graph.predecessors(node) if (u, node) in edges]
        out_es = [edges[(node, v)] for v in graph.successors(node) if (node, v) in edges]
        
        # 允许节点被“抛弃”：入度加总最大为 1
        # 修改：不再强制入等于出，而是流入必须等于流出，但总流入可以是 0（即不选该点）
        problem += pulp.lpSum(in_es) == pulp.lpSum(out_es), f"Flow_{node}"
        problem += pulp.lpSum(in_es) <= 1, f"Limit_{node}"

    problem.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # 5. 结果提取（只保留被选中的边）
    if pulp.LpStatus[problem.status] == "Optimal":
        selected_edges = []
        for (u, v) in edges:
            if u != 'Start' and v != 'End' and edges[(u, v)].varValue > 0.5:
                if 'pos' in graph.nodes[u] and 'pos' in graph.nodes[v]:
                    selected_edges.append((u, v, graph.nodes[u]['pos'], graph.nodes[v]['pos']))
        return selected_edges
    return None

def optimize_and_interpolate_tracks(all_video_tracks, appearance_map, max_time_gap=50, reid_thresh=0.95, spatial_dist_max=25.0):
    """
    升级版：基于时空运动学约束与动态特征衰减的全局轨迹优化系统
    (Spatio-Temporal Kinematic & Dynamic Appearance Tracklet Optimizer)
    """
    # ==========================================
    # 1. 解析与高级轨迹特征提取 (Tracklet Summarization & Kinematic Modeling)
    # ==========================================
    tracklets = {}
    for item in all_video_tracks:
        f_id, t_id = int(item[0]), int(item[1])
        box = np.array(item[2:6], dtype=float)
        score = float(item[6])
        feat = appearance_map.get((f_id, t_id))

        if t_id not in tracklets:
            tracklets[t_id] = {'frames': [], 'boxes': [], 'feats': [], 'scores': []}
        
        tracklets[t_id]['frames'].append(f_id)
        tracklets[t_id]['boxes'].append(box)
        tracklets[t_id]['scores'].append(score)
        if feat is not None:
            tracklets[t_id]['feats'].append(feat)

    t_ids = list(tracklets.keys())
    num_tracks = len(t_ids)
    summaries = []

    for t_id in t_ids:
        data = tracklets[t_id]
        sort_idx = np.argsort(data['frames'])
        frames = np.array(data['frames'])[sort_idx]
        boxes = np.array(data['boxes'])[sort_idx]
        
        # 创新点 1：提取轨迹首尾的瞬时运动学特征 (Velocity Vectors)
        v_tail = np.zeros(2) # 消失前的瞬时速度
        v_head = np.zeros(2) # 出现时的瞬时速度
        if len(frames) > 1:
            # 取最后两帧计算尾部速度，取前两帧计算头部速度
            dt_tail = max(1, frames[-1] - frames[-2])
            v_tail = (boxes[-1][:2] - boxes[-2][:2]) / dt_tail
            
            dt_head = max(1, frames[1] - frames[0])
            v_head = (boxes[1][:2] - boxes[0][:2]) / dt_head
            
        if len(data['feats']) > 0:
            avg_feat = np.mean(np.array(data['feats']), axis=0)
            avg_feat = avg_feat / (np.linalg.norm(avg_feat) + 1e-8) 
        else:
            avg_feat = None
            
        summaries.append({
            'start_f': frames[0], 'end_f': frames[-1],
            'start_box': boxes[0], 'end_box': boxes[-1],
            'v_tail': v_tail, 'v_head': v_head, # 增加运动学状态
            'avg_feat': avg_feat
        })

    # ==========================================
    # 2. 构建多模态时空图代价矩阵 (Multimodal Spatio-Temporal Graph Edge Weights)
    # ==========================================
    cost_matrix = np.full((num_tracks, num_tracks), 1000.0)
    
    # 超参数：可暴露到外部接口进行消融实验
    alpha_decay = 0.01  # 时间衰减系数
    lambda_kinematic = 0.2  # 运动学惩罚项权重
    
    for i in range(num_tracks):
        for j in range(num_tracks):
            if i == j: continue
            
            s_i, s_j = summaries[i], summaries[j]
            dt = s_j['start_f'] - s_i['end_f']
            
            if 0 < dt <= max_time_gap:
                # 2.1 基础外观相似度 (Base Appearance Affinity)
                sim = 0.0
                if s_i['avg_feat'] is not None and s_j['avg_feat'] is not None:
                    sim = np.dot(s_i['avg_feat'], s_j['avg_feat'])
                
                # 创新点 2：时间衰减的外观置信度 (Time-Decayed Affinity)
                # 假设：随着遮挡时间增加，外观发生变化的概率呈指数上升
                decayed_sim = sim * np.exp(-alpha_decay * dt)
                
                # 2.2 空间与运动学特征 (Spatial & Kinematic Features)
                gap_vector = s_j['start_box'][:2] - s_i['end_box'][:2]
                dist = np.linalg.norm(gap_vector)
                
                # 创新点 3：运动方向一致性校验 (Directional Consistency Check)
                # 评估 Tracklet i 的速度方向、Gap 补全方向、Tracklet j 的速度方向是否共线
                kinematic_penalty = 0.0
                if dist > 1e-3:
                    gap_dir = gap_vector / dist
                    # 计算余弦距离作为惩罚：如果夹角大，惩罚高
                    v_i_norm = np.linalg.norm(s_i['v_tail'])
                    if v_i_norm > 1e-3:
                        dir_i = s_i['v_tail'] / v_i_norm
                        kinematic_penalty += (1.0 - np.dot(dir_i, gap_dir)) 
                
                # 2.3 动态权重融合体系 (Adaptive Cost Formulation)
                if sim >= reid_thresh and dist <= spatial_dist_max:
                    # Cost = 外观距离 + 归一化空间距离 + 运动学惩罚
                    cost_matrix[i, j] = (1.0 - decayed_sim) + 0.5 * (dist / spatial_dist_max) + lambda_kinematic * kinematic_penalty

    # ==========================================
    # 3. 全局图匹配 (Global Graph Matching)
    # ==========================================
    # 此时代价矩阵已经蕴含了深度物理规律，直接使用匈牙利算法求解全局最优拓扑
    matches, _, _ = linear_assignment(cost_matrix, thresh=0.8) 
    
    parent = {t_id: t_id for t_id in t_ids}
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    for m in matches:
        r, c = int(m[0]), int(m[1])
        if cost_matrix[r, c] < 1.0: 
            root_r = find(t_ids[r])
            root_c = find(t_ids[c])
            if root_r != root_c:
                parent[root_c] = root_r

    # ==========================================
    # 4. 自适应轨迹重构 (Adaptive Trajectory Reconstruction)
    # ==========================================
    merged_tracks = {}
    for t_id in t_ids:
        root_id = find(t_id)
        if root_id not in merged_tracks:
            merged_tracks[root_id] = {'frames': [], 'boxes': [], 'scores': []}
            
        merged_tracks[root_id]['frames'].extend(tracklets[t_id]['frames'])
        merged_tracks[root_id]['boxes'].extend(tracklets[t_id]['boxes'])
        merged_tracks[root_id]['scores'].extend(tracklets[t_id]['scores'])

    final_results = []
    
    for final_id, data in merged_tracks.items():
        sort_idx = np.argsort(data['frames'])
        f = np.array(data['frames'])[sort_idx]
        b = np.array(data['boxes'])[sort_idx]
        s = np.array(data['scores'])[sort_idx]
        
        # 无损写入原始帧 (Preserve High-Confidence Online Output)
        for i in range(len(f)):
            final_results.append([f[i], final_id, b[i][0], b[i][1], b[i][2], b[i][3], s[i]])
            
        # 安全区间插值平滑 (Safe-zone Interpolation Smoothing)
        for k in range(len(f) - 1):
            gap = f[k+1] - f[k]
            if 1 < gap <= 10: 
                for step in range(1, gap):
                    ratio = step / gap
                    interp_box = b[k] + ratio * (b[k+1] - b[k])
                    interp_score = s[k] + ratio * (s[k+1] - s[k]) 
                    final_results.append([f[k] + step, final_id, interp_box[0], interp_box[1], interp_box[2], interp_box[3], interp_score])

    return final_results

# def optimize_and_interpolate_tracks_1(all_video_tracks, appearance_map, max_time_gap=70, reid_thresh=0.85, spatial_dist_max=25.0):
#     # ==========================================
#     # 1. 解析与高级轨迹特征提取 (Tracklet Summarization)
#     # ==========================================
#     tracklets = {}
#     for item in all_video_tracks:
#         f_id, t_id = int(item[0]), int(item[1])
#         # 假设原始输入框格式为 [x1, y1, x2, y2]
#         box = np.array(item[2:6], dtype=float) 
#         score = float(item[6])
#         feat = appearance_map.get((f_id, t_id))

#         if t_id not in tracklets:
#             tracklets[t_id] = {'frames': [], 'boxes': [], 'feats': [], 'scores': []}
        
#         tracklets[t_id]['frames'].append(f_id)
#         tracklets[t_id]['boxes'].append(box)
#         tracklets[t_id]['scores'].append(score)
#         if feat is not None:
#             tracklets[t_id]['feats'].append(feat)

#     t_ids = list(tracklets.keys())
#     num_tracks = len(t_ids)
#     summaries = []

#     for t_id in t_ids:
#         data = tracklets[t_id]
#         sort_idx = np.argsort(data['frames'])
#         frames = np.array(data['frames'])[sort_idx]
#         boxes = np.array(data['boxes'])[sort_idx]
        
#         # 将 [x1, y1, x2, y2] 转换为中心点与尺度: [cx, cy, w, h]
#         centers = np.array([[(b[0]+b[2])/2.0, (b[1]+b[3])/2.0] for b in boxes])
#         scales = np.array([[b[2]-b[0], b[3]-b[1]] for b in boxes])
        
#         v_tail = np.zeros(2)
#         v_head = np.zeros(2)
        
#         if len(frames) > 1:
#             # 使用中心点计算速度，更能反映真实的物理运动
#             dt_tail = max(1, frames[-1] - frames[-2])
#             v_tail = (centers[-1] - centers[-2]) / dt_tail
            
#             dt_head = max(1, frames[1] - frames[0])
#             v_head = (centers[1] - centers[0]) / dt_head
            
#         if len(data['feats']) > 0:
#             avg_feat = np.mean(np.array(data['feats']), axis=0)
#             avg_feat = avg_feat / (np.linalg.norm(avg_feat) + 1e-8) 
#         else:
#             avg_feat = None
            
#         summaries.append({
#             'start_f': frames[0], 'end_f': frames[-1],
#             'start_center': centers[0], 'end_center': centers[-1],
#             'start_scale': scales[0], 'end_scale': scales[-1], # 新增尺度记录
#             'v_tail': v_tail, 'v_head': v_head,
#             'avg_feat': avg_feat
#         })

#     # ==========================================
#     # 2. 构建多模态时空图代价矩阵 
#     # ==========================================
#     cost_matrix = np.full((num_tracks, num_tracks), 1000.0)
    
#     # 超参数体系
#     alpha_decay = 0.01      # 时间衰减系数
#     lambda_kinematic = 0.3  # 运动学惩罚项权重
#     lambda_scale = 0.2      # 尺度惩罚项权重
    
#     for i in range(num_tracks):
#         for j in range(num_tracks):
#             if i == j: continue
            
#             s_i, s_j = summaries[i], summaries[j]
#             dt = s_j['start_f'] - s_i['end_f']
            
#             if 0 < dt <= max_time_gap:
#                 # 2.1 基础外观与时间衰减
#                 sim = 0.0
#                 if s_i['avg_feat'] is not None and s_j['avg_feat'] is not None:
#                     sim = np.dot(s_i['avg_feat'], s_j['avg_feat'])
#                 decayed_sim = sim * np.exp(-alpha_decay * dt)
                
#                 # 2.2 空间中心点距离
#                 gap_vector = s_j['start_center'] - s_i['end_center']
#                 dist = np.linalg.norm(gap_vector)
                
#                 # 2.3 升级点一：双向运动学一致性 (Bidirectional Kinematic Consistency)
#                 kinematic_penalty = 0.0
#                 if dist > 1e-3:
#                     gap_dir = gap_vector / dist
                    
#                     # 惩罚项 1：Track i 消失前的运动方向应指向 Gap
#                     v_i_norm = np.linalg.norm(s_i['v_tail'])
#                     if v_i_norm > 1e-3:
#                         dir_i = s_i['v_tail'] / v_i_norm
#                         kinematic_penalty += (1.0 - np.dot(dir_i, gap_dir)) 
                        
#                     # 惩罚项 2：Track j 出现时的运动方向应顺延自 Gap (双向约束)
#                     v_j_norm = np.linalg.norm(s_j['v_head'])
#                     if v_j_norm > 1e-3:
#                         dir_j = s_j['v_head'] / v_j_norm
#                         kinematic_penalty += (1.0 - np.dot(dir_j, gap_dir))
                
#                 # 2.4 升级点二：尺度感知惩罚 (Scale-Aware Penalty)
#                 # 计算目标遮挡前后的宽高变化率，归一化到 0~2 之间
#                 scale_i = s_i['end_scale'] # [w, h]
#                 scale_j = s_j['start_scale'] # [w, h]
#                 scale_penalty = np.sum(np.abs(scale_i - scale_j) / np.maximum(scale_i, scale_j))
                
#                 # 2.5 代价融合
#                 if sim >= reid_thresh and dist <= spatial_dist_max:
#                     cost_matrix[i, j] = (1.0 - decayed_sim) + \
#                                         0.5 * (dist / spatial_dist_max) + \
#                                         lambda_kinematic * kinematic_penalty + \
#                                         lambda_scale * scale_penalty
                    
#     # ==========================================
#     # 3. 全局图匹配 (Global Graph Matching)
#     # ==========================================
#     # 此时代价矩阵已经蕴含了深度物理规律，直接使用匈牙利算法求解全局最优拓扑
#     matches, _, _ = linear_assignment(cost_matrix, thresh=0.8) 
    
#     parent = {t_id: t_id for t_id in t_ids}
#     def find(i):
#         if parent[i] == i: return i
#         parent[i] = find(parent[i])
#         return parent[i]

#     for m in matches:
#         r, c = int(m[0]), int(m[1])
#         if cost_matrix[r, c] < 1.0: 
#             root_r = find(t_ids[r])
#             root_c = find(t_ids[c])
#             if root_r != root_c:
#                 parent[root_c] = root_r

#     # ==========================================
#     # 4. 自适应轨迹重构 (Adaptive Trajectory Reconstruction)
#     # ==========================================
#     merged_tracks = {}
#     for t_id in t_ids:
#         root_id = find(t_id)
#         if root_id not in merged_tracks:
#             merged_tracks[root_id] = {'frames': [], 'boxes': [], 'scores': []}
            
#         merged_tracks[root_id]['frames'].extend(tracklets[t_id]['frames'])
#         merged_tracks[root_id]['boxes'].extend(tracklets[t_id]['boxes'])
#         merged_tracks[root_id]['scores'].extend(tracklets[t_id]['scores'])

#     final_results = []
    
#     for final_id, data in merged_tracks.items():
#         sort_idx = np.argsort(data['frames'])
#         f = np.array(data['frames'])[sort_idx]
#         b = np.array(data['boxes'])[sort_idx]
#         s = np.array(data['scores'])[sort_idx]
        
#         # 无损写入原始帧 (Preserve High-Confidence Online Output)
#         for i in range(len(f)):
#             final_results.append([f[i], final_id, b[i][0], b[i][1], b[i][2], b[i][3], s[i]])
            
#         # 安全区间插值平滑 (Safe-zone Interpolation Smoothing)
#         for k in range(len(f) - 1):
#             gap = f[k+1] - f[k]
#             if 1 < gap <= 10: 
#                 for step in range(1, gap):
#                     ratio = step / gap
#                     interp_box = b[k] + ratio * (b[k+1] - b[k])
#                     interp_score = s[k] + ratio * (s[k+1] - s[k]) 
#                     final_results.append([f[k] + step, final_id, interp_box[0], interp_box[1], interp_box[2], interp_box[3], interp_score])

#     return final_results

import numpy as np

def optimize_and_interpolate_tracks_1(
        all_video_tracks,
        appearance_map,
        max_time_gap=70,
        reid_thresh=0.85,
        spatial_dist_max=25.0,

        # ===== Ablation Switches =====
        use_temporal_decay=True,
        use_spatial_cost=True,
        use_kinematic_cost=True,
        use_scale_cost=True,
        use_interpolation=True,
):
    """
    Global Optimization with Ablation Switches

    Default:
        All switches=True
        ==> Exactly equivalent to the original implementation.
    """

    # ==========================================================
    # 1. Tracklet Summarization
    # ==========================================================
    tracklets = {}

    for item in all_video_tracks:

        f_id = int(item[0])
        t_id = int(item[1])

        box = np.array(item[2:6], dtype=float)
        score = float(item[6])

        feat = appearance_map.get((f_id, t_id))

        if t_id not in tracklets:

            tracklets[t_id] = {
                'frames': [],
                'boxes': [],
                'feats': [],
                'scores': []
            }

        tracklets[t_id]['frames'].append(f_id)
        tracklets[t_id]['boxes'].append(box)
        tracklets[t_id]['scores'].append(score)

        if feat is not None:
            tracklets[t_id]['feats'].append(feat)

    t_ids = list(tracklets.keys())

    num_tracks = len(t_ids)

    summaries = []

    for t_id in t_ids:

        data = tracklets[t_id]

        sort_idx = np.argsort(data['frames'])

        frames = np.array(data['frames'])[sort_idx]

        boxes = np.array(data['boxes'])[sort_idx]

        centers = np.array([
            [
                (b[0] + b[2]) / 2.0,
                (b[1] + b[3]) / 2.0
            ]
            for b in boxes
        ])

        scales = np.array([
            [
                b[2] - b[0],
                b[3] - b[1]
            ]
            for b in boxes
        ])

        v_tail = np.zeros(2)

        v_head = np.zeros(2)

        if len(frames) > 1:

            dt_tail = max(1, frames[-1] - frames[-2])

            v_tail = (centers[-1] - centers[-2]) / dt_tail

            dt_head = max(1, frames[1] - frames[0])

            v_head = (centers[1] - centers[0]) / dt_head

        if len(data['feats']) > 0:

            avg_feat = np.mean(
                np.array(data['feats']),
                axis=0
            )

            avg_feat = avg_feat / (
                    np.linalg.norm(avg_feat) + 1e-8
            )

        else:

            avg_feat = None

        summaries.append({

            'start_f': frames[0],
            'end_f': frames[-1],

            'start_center': centers[0],
            'end_center': centers[-1],

            'start_scale': scales[0],
            'end_scale': scales[-1],

            'v_tail': v_tail,
            'v_head': v_head,

            'avg_feat': avg_feat

        })

    # ==========================================================
    # 2. Cost Matrix
    # ==========================================================

    cost_matrix = np.full(
        (num_tracks, num_tracks),
        1000.0
    )

    alpha_decay = 0.01

    lambda_kinematic = 0.3

    lambda_scale = 0.2

    for i in range(num_tracks):

        for j in range(num_tracks):

            if i == j:
                continue

            s_i = summaries[i]
            s_j = summaries[j]

            dt = s_j['start_f'] - s_i['end_f']

            if not (0 < dt <= max_time_gap):
                continue

            ####################################################
            # Appearance
            ####################################################

            sim = 0.0

            if (
                s_i['avg_feat'] is not None
                and
                s_j['avg_feat'] is not None
            ):

                sim = np.dot(
                    s_i['avg_feat'],
                    s_j['avg_feat']
                )

            ####################################################
            # Temporal Decay
            ####################################################

            if use_temporal_decay:

                decayed_sim = (
                    sim *
                    np.exp(-alpha_decay * dt)
                )

            else:

                decayed_sim = sim

            ####################################################
            # Spatial Distance
            ####################################################

            gap_vector = (
                s_j['start_center']
                -
                s_i['end_center']
            )

            dist = np.linalg.norm(
                gap_vector
            )

            ####################################################
            # Candidate Generation
            # !!! NEVER CHANGE !!!
            ####################################################

            if sim < reid_thresh:
                continue

            if dist > spatial_dist_max:
                continue

            ####################################################
            # Cost Initialization
            ####################################################

            cost = 1.0 - decayed_sim

            ####################################################
            # Spatial Cost
            ####################################################

            if use_spatial_cost:

                cost += (
                        0.5 *
                        (
                                dist /
                                spatial_dist_max
                        )
                )

            ####################################################
            # Kinematic Cost
            ####################################################

            if use_kinematic_cost:

                kinematic_penalty = 0.0

                if dist > 1e-3:

                    gap_dir = gap_vector / dist

                    v_i_norm = np.linalg.norm(
                        s_i['v_tail']
                    )

                    if v_i_norm > 1e-3:

                        dir_i = (
                                s_i['v_tail']
                                /
                                v_i_norm
                        )

                        kinematic_penalty += (
                                1.0 -
                                np.dot(
                                    dir_i,
                                    gap_dir
                                )
                        )

                    v_j_norm = np.linalg.norm(
                        s_j['v_head']
                    )

                    if v_j_norm > 1e-3:

                        dir_j = (
                                s_j['v_head']
                                /
                                v_j_norm
                        )

                        kinematic_penalty += (
                                1.0 -
                                np.dot(
                                    dir_j,
                                    gap_dir
                                )
                        )

                cost += (
                        lambda_kinematic *
                        kinematic_penalty
                )

            ####################################################
            # Scale Cost
            ####################################################

            if use_scale_cost:

                scale_i = s_i['end_scale']

                scale_j = s_j['start_scale']

                scale_penalty = np.sum(

                    np.abs(
                        scale_i - scale_j
                    )

                    /

                    np.maximum(
                        scale_i,
                        scale_j
                    )

                )

                cost += (
                        lambda_scale *
                        scale_penalty
                )

            ####################################################
            # Save Cost
            ####################################################

            cost_matrix[i, j] = cost
                # ==========================================================
    # 3. Global Graph Matching
    # ==========================================================

    matches, _, _ = linear_assignment(
        cost_matrix,
        thresh=0.8
    )

    # ==========================================================
    # Union-Find
    # ==========================================================

    parent = {
        t_id: t_id
        for t_id in t_ids
    }

    def find(i):

        if parent[i] == i:
            return i

        parent[i] = find(parent[i])

        return parent[i]

    ############################################################

    for m in matches:

        r = int(m[0])
        c = int(m[1])

        if cost_matrix[r, c] < 1.0:

            root_r = find(
                t_ids[r]
            )

            root_c = find(
                t_ids[c]
            )

            if root_r != root_c:

                parent[root_c] = root_r

    # ==========================================================
    # 4. Merge Tracklets
    # ==========================================================

    merged_tracks = {}

    for t_id in t_ids:

        root_id = find(t_id)

        if root_id not in merged_tracks:

            merged_tracks[root_id] = {

                'frames': [],
                'boxes': [],
                'scores': []

            }

        merged_tracks[root_id]['frames'].extend(
            tracklets[t_id]['frames']
        )

        merged_tracks[root_id]['boxes'].extend(
            tracklets[t_id]['boxes']
        )

        merged_tracks[root_id]['scores'].extend(
            tracklets[t_id]['scores']
        )

    # ==========================================================
    # 5. Reconstruction
    # ==========================================================

    final_results = []

    for final_id, data in merged_tracks.items():

        sort_idx = np.argsort(
            data['frames']
        )

        f = np.array(
            data['frames']
        )[sort_idx]

        b = np.array(
            data['boxes']
        )[sort_idx]

        s = np.array(
            data['scores']
        )[sort_idx]

        ########################################################
        # Preserve Original Detections
        ########################################################

        for i in range(len(f)):

            final_results.append([

                f[i],
                final_id,

                b[i][0],
                b[i][1],
                b[i][2],
                b[i][3],

                s[i]

            ])

        ########################################################
        # Interpolation Ablation
        ########################################################

        if use_interpolation:

            for k in range(len(f) - 1):

                gap = f[k + 1] - f[k]

                if 1 < gap <= 10:

                    for step in range(1, gap):

                        ratio = step / gap

                        interp_box = (

                            b[k]
                            +
                            ratio *
                            (
                                b[k + 1]
                                -
                                b[k]
                            )

                        )

                        interp_score = (

                            s[k]
                            +
                            ratio *
                            (
                                s[k + 1]
                                -
                                s[k]
                            )

                        )

                        final_results.append([

                            f[k] + step,

                            final_id,

                            interp_box[0],
                            interp_box[1],
                            interp_box[2],
                            interp_box[3],

                            interp_score

                        ])

    # ==========================================================
    # Sort
    # ==========================================================

    final_results = sorted(

        final_results,

        key=lambda x: (
            x[0],
            x[1]
        )

    )

    return final_results

def build_global_graph(all_video_tracks, raw_detections):
    """
    将 Fasttracker 轨迹和未匹配的原始检测结果融合成一张全局图。
    
    参数:
    all_video_tracks: list, Fasttracker的输出轨迹
    raw_detections: dict, 按帧组织的原始检测结果 {frame_id: [[x1, y1, w, h, score], ...]}
    """
    G = nx.DiGraph()
    tracks_by_id = {}
    
    # 记录每帧已经被 tracker 占用的检测框，用于后续去重
    occupied_boxes_per_frame = {}
    
    # ==========================================
    # 第一步：处理强约束节点 (Fasttracker 结果)
    # ==========================================
    for item in all_video_tracks:
        frame_id = int(item[0])
        track_id = int(item[1])
        x1, y1, w, h = float(item[2]), float(item[3]), float(item[4]), float(item[5])
        tlwh = np.array([x1, y1, w, h], dtype=np.float32)
        score = float(item[6])
        
        cx, cy = x1 + w / 2.0, y1 + h / 2.0
        pos_info = (cx, cy, w, h)
        
        node_id = f"T_{track_id}_{frame_id}" # 前缀 T 表示 Track 节点
        
        G.add_node(node_id, 
                   track_id=track_id, 
                   frame_id=frame_id, 
                   bbox=tlwh, 
                   score=score,
                   pos=pos_info,
                   node_type='track') 
        
        if track_id not in tracks_by_id:
            tracks_by_id[track_id] = []
        tracks_by_id[track_id].append((frame_id, node_id))
        
        if frame_id not in occupied_boxes_per_frame:
            occupied_boxes_per_frame[frame_id] = []
        occupied_boxes_per_frame[frame_id].append(tlwh)

    # ==========================================
    # 第二步：处理弱约束节点 (孤立的 Detection)
    # ==========================================
    det_nodes_by_frame = {} # 方便后续按帧连边
    
    det_id_counter = 0
    for frame_id, dets in raw_detections.items():
        det_nodes_by_frame[frame_id] = []
        occupied_boxes = occupied_boxes_per_frame.get(frame_id, [])
        
        for det in dets:
            x1, y1, w, h = det[0], det[1], det[2], det[3]
            score = det[4] if len(det) > 4 else 1.0
            tlwh = np.array([x1, y1, w, h], dtype=np.float32)
            
            # 判断是否与现有 tracker 框重合 (IoU > 0.5 视为同一个目标)
            is_occupied = any(compute_iou(tlwh, occ_box) > 0.9 for occ_box in occupied_boxes)
            
            if not is_occupied and score > 0.6: # 设定一个检测置信度阈值过滤噪点
                cx, cy = x1 + w / 2.0, y1 + h / 2.0
                pos_info = (cx, cy, w, h)
                node_id = f"D_{det_id_counter}_{frame_id}" # 前缀 D 表示 Detection 节点
                det_id_counter += 1
                
                G.add_node(node_id, 
                           track_id=-1, # -1 表示尚未分配 ID
                           frame_id=frame_id, 
                           bbox=tlwh, 
                           score=score,
                           pos=pos_info,
                           node_type='det')
                
                det_nodes_by_frame[frame_id].append(node_id)

    # ==========================================
    # 第三步：建立强约束边 (轨迹内部相连)
    # ==========================================
    for t_id, nodes in tracks_by_id.items():
        nodes.sort(key=lambda x: x[0])
        for i in range(len(nodes) - 1):
            u_frame, u_node = nodes[i]
            v_frame, v_node = nodes[i+1]
            time_gap = v_frame - u_frame
            
            if 0 < time_gap <= 30: 
                # 强约束边赋予极高权重，确保 ILP 优先选择
                G.add_edge(u_node, v_node, edge_type='baseline', weight=50.0) 

    # ==========================================
    # 第四步：建立弱约束边 (时空域内的候选连接)
    # ==========================================
    # 获取所有按帧排序的节点集合（包含 T 节点和 D 节点）
    all_frames = sorted(list(set([data['frame_id'] for n, data in G.nodes(data=True)])))
    nodes_per_frame = {f: [] for f in all_frames}
    for n, data in G.nodes(data=True):
        nodes_per_frame[data['frame_id']].append(n)
        
    MAX_TIME_GAP = 1       # 允许连接的最大帧间距
    MAX_SPATIAL_DIST = 100  # 允许连接的最大像素距离 (根据你的画面分辨率调整)
    
    for i, u_frame in enumerate(all_frames):
        u_nodes = nodes_per_frame[u_frame]
        
        # 向后搜索一定时间窗口内的帧
        for v_frame in all_frames[i+1:]:
            time_gap = v_frame - u_frame
            if time_gap > MAX_TIME_GAP:
                break # 超出时间窗口，停止向后搜索
                
            v_nodes = nodes_per_frame[v_frame]
            
            for u in u_nodes:
                for v in v_nodes:
                    # 如果 u 和 v 都是同一条强约束轨迹内的节点，跳过（强约束边已建好）
                    u_data, v_data = G.nodes[u], G.nodes[v]
                    if u_data['node_type'] == 'track' and v_data['node_type'] == 'track':
                        if u_data['track_id'] == v_data['track_id']:
                            continue
                    
                    # 提取中心点计算欧氏距离
                    u_cx, u_cy = u_data['pos'][0], u_data['pos'][1]
                    v_cx, v_cy = v_data['pos'][0], v_data['pos'][1]
                    dist = math.hypot(v_cx - u_cx, v_cy - u_cy)
                    
                    # 运动学约束过滤：如果速度/距离合理，则建立弱约束边
                    max_dist_allowed = 40.0 + (time_gap * 15.0) 
                    if dist < max_dist_allowed:
                        edge_weight = 12.0 - (1.2 * time_gap) - (0.04 * dist)
                        if edge_weight > 0:
                            G.add_edge(u, v, edge_type='hypothesis', weight=edge_weight)

    print(f"全局图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")
    return G

def build_graph_from_fasttracker(all_video_tracks):
    """
    将 Fasttracker 的跟踪结果转换为有向图。
    【修复】：pos 属性现在包含 (cx, cy, w, h) 以供解析函数计算左上角坐标。
    """
    G = nx.DiGraph()
    
    tracks_by_id = {}
    
    for item in all_video_tracks:
        frame_id = int(item[0])
        track_id = int(item[1])
        
        # MOT格式: [frame, id, x1, y1, w, h, score, ...]
        x1, y1, w, h = float(item[2]), float(item[3]), float(item[4]), float(item[5])
        tlwh = np.array([x1, y1, w, h], dtype=np.float32)
        score = float(item[6])
        
        # --- [关键修改] pos 必须包含 (cx, cy, w, h) ---
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        
        # 之前是 (frame_id, cx, cy) -> 导致了 IndexError
        # 现在改为 (cx, cy, w, h) -> 满足 parse_prediction_fast 的索引需求
        pos_info = (cx, cy, w, h) 

        # 1. 唯一标识节点 (字符串)
        node_id = f"{track_id}_{frame_id}"
        
        # 2. 添加节点
        # 注意：我们在节点属性里也存一个 frame_id (int)，方便后续万一需要查属性
        G.add_node(node_id, 
                   track_id=track_id, 
                   frame_id=frame_id, 
                   bbox=tlwh, 
                   score=score,
                   pos=pos_info) 
        
        if track_id not in tracks_by_id:
            tracks_by_id[track_id] = []
        tracks_by_id[track_id].append((frame_id, node_id))

    # 3. 建立轨迹内部的边
    for t_id, nodes in tracks_by_id.items():
        nodes.sort(key=lambda x: x[0])
        for i in range(len(nodes) - 1):
            u_frame, u_node = nodes[i]
            v_frame, v_node = nodes[i+1]
            
            time_gap = v_frame - u_frame
            if 0 < time_gap <= 30: 
                # 权重逻辑保持不变
                edge_weight = 10.0 - 0.1 * time_gap 
                G.add_edge(u_node, v_node, edge_type='baseline', weight=edge_weight) 
                           
    print(f"图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条强约束边")
    return G



def solve_ilp_f(graph, optim_config = None):
    import pulp
    import numpy as np

    # start_cost = optim_config.get('start_cost', 0.0)
    # end_cost = optim_config.get('end_cost', 0.0)
    # min_track_length = optim_config.get('min_track_length', None)
    # max_track_length = optim_config.get('max_track_length', None)

    # density_threshold = optim_config.get('density_threshold', 0.5)
    # min_distance_threshold = optim_config.get('min_distance_threshold', 3.0)

    start_cost = 20.0
    end_cost = 0
    min_track_length = 1
    max_track_length = None
    density_threshold = 1.0
    min_distance_threshold = 0.0

    problem = pulp.LpProblem("TrackForming", pulp.LpMaximize)

    edges = {}
    positions = {}
    nodes = list(graph.nodes())

    for u, v in graph.edges():
        var_name = f"edge_{u}_{v}"
        edges[(u, v)] = pulp.LpVariable(var_name, cat=pulp.LpBinary)
        positions[u] = graph.nodes[u]['pos']
        positions[v] = graph.nodes[v]['pos']

    if start_cost > 0 or end_cost > 0:
        start_node = 'Start'
        end_node = 'End'
        graph.add_node(start_node)
        graph.add_node(end_node)
        for node in nodes:
            var_start = f"edge_{start_node}_{node}"
            var_end = f"edge_{node}_{end_node}"
            edges[(start_node, node)] = pulp.LpVariable(var_start, cat=pulp.LpBinary)
            edges[(node, end_node)] = pulp.LpVariable(var_end, cat=pulp.LpBinary)
            graph.add_edge(start_node, node, weight=start_cost)
            graph.add_edge(node, end_node, weight=end_cost)

    problem += pulp.lpSum(edges[e] * graph.edges[e]['weight'] for e in edges)

    # 入出度限制
    for node in nodes:
        incoming_edges = [edges[(u, node)] for u in graph.predecessors(node) if (u, node) in edges]
        outgoing_edges = [edges[(node, v)] for v in graph.successors(node) if (node, v) in edges]
        problem += pulp.lpSum(incoming_edges) <= 1, f"MaxIn_{node}"
        problem += pulp.lpSum(outgoing_edges) <= 1, f"MaxOut_{node}"

    if min_track_length is not None or max_track_length is not None:
        track_vars = {}
        track_id = 0

        def add_track_length_constraints(current_node, current_length):
            outgoing_edges = [edges[(current_node, v)] for v in graph.successors(current_node) if (current_node, v) in edges]
            problem += track_var >= current_length, f"TrackLength_{current_node}"
            for edge_var, successor_node in zip(outgoing_edges, graph.successors(current_node)):
                problem += edge_var <= track_var, f"EdgeTrack_{current_node}_{successor_node}"
                add_track_length_constraints(successor_node, current_length + 1)

        for node in nodes:
            incoming_edges = [edges[(u, node)] for u in graph.predecessors(node) if (u, node) in edges]
            if len(incoming_edges) == 0:
                track_var = pulp.LpVariable(f"Track_{track_id}", lowBound=0, cat=pulp.LpInteger)
                track_id += 1
                add_track_length_constraints(node, 1)
                if min_track_length is not None:
                    problem += track_var >= min_track_length, f"MinTrackLength_{node}"
                if max_track_length is not None:
                    problem += track_var <= max_track_length, f"MaxTrackLength_{node}"

    problem.solve()

    if pulp.LpStatus[problem.status] == "Optimal":
        # solution_edges: list of (u,v,pos_u,pos_v)
        selected_edges = [
            (u, v, positions[u], positions[v])
            for u, v in edges
            if edges[(u, v)].varValue > 0.5 and u in positions and v in positions
            if u != 'Start' and v != 'End'
        ]

        # 构建轨迹(含edges)以进行密度检查
        trajectories_with_edges = build_trajectories_with_edges(selected_edges)

        filtered_edges = []
        for traj_nodes, traj_positions, traj_edges in trajectories_with_edges:
            density = compute_track_density(traj_positions)
            total_displacement = np.linalg.norm(traj_positions[-1] - traj_positions[0])
            if density <= density_threshold and total_displacement >= min_distance_threshold:
                # 保留该轨迹的所有edges
                filtered_edges.extend(traj_edges)

        return filtered_edges
    else:
        return None

def solve_ilp_fast(graph):
    """
    修复后的 ILP 求解器
    逻辑说明：
    - 每个节点 n 对应一个二进制变量 z_n，表示该点是否被包含在轨迹中。
    - 目标函数：Maximize sum(Edge_weight * x_uv) + sum(z_n * start_cost)
    """
    start_cost = -2.0
    end_cost = 0.1
    
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
    
def parse_prediction_fast(solution_edges):
    """ Parses the solution edges to create tracks with consistent IDs."""
    node_successors = {}
    node_predecessors = {}
    nodes = set()
    node_centers = {} # 这里存的其实是 (cx, cy, w, h)
    
    for u, v, pos_u, pos_v in solution_edges:
        node_successors[u] = v
        node_predecessors[v] = u
        nodes.add(u)
        nodes.add(v)
        node_centers[u] = pos_u
        node_centers[v] = pos_v

    # 查找起始节点
    start_nodes = [node for node in nodes if node not in node_predecessors]

    pred_tracks = {}
    
    # 这里的 frame_tracks 如果不需要在这个函数里直接生成，可以不写，
    # 但为了兼容你的逻辑，我们保留它
    frame_tracks = {} 
    
    track_id_counter = 0
    visited_nodes = set()

    for start_node in start_nodes:
        node = start_node
        track_id = track_id_counter
        track_id_counter += 1
        
        while True:
            if node in visited_nodes:
                break
            visited_nodes.add(node)
            
            # --- [关键修改] 从 node_id 字符串解析 frame_id ---
            # node 格式为 "trackID_frameID"，我们需要提取 frameID
            try:
                # 假设 node_id 格式是 "originalID_frameID"
                # split('_')[-1] 取最后一部分作为 frame_id
                frame_id = int(str(node).split('_')[-1])
            except ValueError:
                print(f"Warning: Could not parse frame_id from node {node}")
                frame_id = -1

            center = node_centers.get(node) # (cx, cy, w, h)
            if center is None:
                break
            
            # center[0]=cx, center[1]=cy, center[2]=w, center[3]=h
            tl_x = center[0] - center[2] / 2
            tl_y = center[1] - center[3] / 2
            
            # 更新 pred_tracks
            if track_id not in pred_tracks:
                pred_tracks[track_id] = {}
            
            # 存储格式: (x, y, w, h, is_virtual)
            pred_tracks[track_id][frame_id] = (tl_x, tl_y, center[2], center[3], False)
            
            # 移动到下一个节点
            if node in node_successors:
                node = node_successors[node]
            else:
                break
                
    return pred_tracks, frame_tracks

def rebuild_frame_tracks_from_pred_tracks_fast(pred_tracks):
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

def convert_frame_tracks_to_mot_eval_fast(frame_tracks):
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

def insert_virtual_nodes_fast(pred_tracks_initial,  # unused, just keep signature
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
    # max_age = optim_config.get('max_virtual', 10)
    # iou_threshold = optim_config.get('iou_threshold', 0.3)
    max_age =  10
    iou_threshold = 0.3

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

def post_process_tracks_fast(pred_tracks, optim_config = None):
    """
    1) 过滤真实节点数 < min_track_length 的轨迹
    2) 保留虚拟节点，以供可视化或评估
    假设 pred_tracks[tid][frame] 可能是:
      (x, y, w, h, is_virtual) 或 (x, y, w, h, is_virtual, vx, vy)
    """
    # min_track_length = optim_config.get('min_track_length', 2)
    min_track_length = 2
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
    import numpy as np
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