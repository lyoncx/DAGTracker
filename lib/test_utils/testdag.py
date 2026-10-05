from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import sys
import os
# Unused in the active runner; the Optuna block below is retained as a commented
# experiment branch for later parameter-search work.
# import optuna
# 获取项目根目录路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本所在目录（scripts）
project_root = os.path.dirname(current_script_dir)  # 项目根目录

import matplotlib
matplotlib.use('Agg')  # 强制使用无界面后端
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# 将项目根目录添加到系统路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import itertools


from lib.models.stNet import get_det_net, load_model, save_model
from lib.dataset.dataset_factory import get_dataset
from lib.utils_eval.evaluation_final_func import eval_func_final
from lib.test_utils.show_imgs import *
from lib.test_utils.process_img_dets import *
from lib.tracker.fasttracker import Fasttracker
from lib.utils_eval.eval import *
from lib.tracker.global_track import *
from lib.tracker.global_opm import *
import GPUtil
# Used only by the commented MAT-export branch near the inference loop.
# import scipy.io as scio
import datetime


def extract_online_features_for_offline_dets(offline_dets, feature_map, original_img_shape):
    """
    使用在线生成的特征图(feature_map)为离线检测框(offline_dets)提取外观特征。
    
    Args:
        offline_dets: numpy array [N, 5] (x1, y1, x2, y2, conf)
        feature_map: torch.Tensor [1, C, H_feat, W_feat]
        original_img_shape: 原始图像的分辨率 (H, W)，用于坐标缩放映射
    """
    h_feat, w_feat = feature_map.shape[2:]
    h_orig, w_orig = original_img_shape
    
    f1 = []
    # 确保 feature_map 在正确的设备上
    device = feature_map.device
    feat_dim = feature_map.size(1)

    if offline_dets is None or len(offline_dets) == 0:
        return {'1': np.array([]), 'f1': [], 'scores': np.array([])}

    for box in offline_dets:
        x_min, y_min, x_max, y_max, conf = box

        # 1. 将离线框坐标从原始图像尺度映射到特征图尺度
        # 注意：这里需要根据你模型训练时的预处理逻辑（如 padding 或 resize）进行微调
        x_min_f = int(max(0, x_min * w_feat / w_orig))
        x_max_f = int(min(w_feat, x_max * w_feat / w_orig))
        y_min_f = int(max(0, y_min * h_feat / h_orig))
        y_max_f = int(min(h_feat, y_max * h_feat / h_orig))

        # 2. 裁剪与池化
        if x_max_f <= x_min_f or y_max_f <= y_min_f:
            f1.append(torch.zeros(feat_dim).to(device))
            continue

        cropped_feature = feature_map[:, :, y_min_f:y_max_f, x_min_f:x_max_f]
        
        if cropped_feature.numel() == 0:
            feature_vector = torch.zeros(feat_dim).to(device)
        else:
            # 使用自适应平均池化将区域压缩为 1x1
            pooled_feature = F.adaptive_avg_pool2d(cropped_feature, (1, 1))
            feature_vector = pooled_feature.view(-1)
        
        f1.append(feature_vector)

    # 返回符合跟踪器要求的 ret 结构
    ret = {
        1: offline_dets,           # 键为整数 1，存储坐标
        'f1': f1,                  # 在线提取的特征列表 (List of Tensors)
        'scores': offline_dets[:, 4],
        'clses': np.ones(len(offline_dets))
    }
    return ret

# 1. 确保在循环开始前加载离线检测数据
def load_offline_dets(txt_path):
    """
    适配你保存的 TXT 格式: 
    frame_id, center_x, center_y, width, height, conf, -1, -1, -1
    返回字典 {frame_id: [[x1, y1, x2, y2, conf], ...]}
    """
    offline_data = {}
    if not os.path.exists(txt_path):
        print(f"警告：找不到离线检测文件 {txt_path}")
        return offline_data
        
    try:
        # 使用 numpy 加载数据
        data = np.loadtxt(txt_path, delimiter=',')
        if data.size == 0:
            return offline_data
        
        # 处理单行数据的情况
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        for row in data:
            f_id = int(row[0])
            cx, cy, w, h, conf = row[1:6] # 根据你保存的代码索引：1:cx, 2:cy, 3:w, 4:h, 5:conf
            
            # 【核心转换】：从中心点+宽高转回 左上角+右下角
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            
            if f_id not in offline_data:
                offline_data[f_id] = []
            offline_data[f_id].append([x1, y1, x2, y2, conf])
            
    except Exception as e:
        print(f"解析检测文件 {txt_path} 出错: {e}")
        
    return {k: np.array(v) for k, v in offline_data.items()}

def save_offline_cache(video_id, det_results, tracklets, raw_dets, cache_dir):
    """
    保存单个视频的检测和跟踪缓存
    det_results: {img_path: {'bbox':..., 'feat':..., 'conf':...}}
    tracklets: [[frame, id, x, y, w, h, conf, ...], ...]
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{video_id}_cache.pkl")
    data = {
        'det_results': det_results,
        'tracklets': tracklets,
        'raw_dets': raw_dets
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Successfully cached offline data for {video_id} to {cache_path}")

def load_offline_cache(video_id, cache_dir):
    cache_path = os.path.join(cache_dir, f"{video_id}_cache.pkl")
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

def run_alignment_debug(video_id, structured_tracklets, cells_by_t, scale_factor=0.4128):
    """
    针对 List[Dict] 结构的修正版诊断函数
    structured_tracklets: [{'frames': [...], 'bboxes': [[x1,y1,x2,y2],...]}, ...]
    """
    print(f"\n[开始深度诊断] 视频 ID: {video_id}")
    
    # 1. 寻找一个共有的帧作为分析样本
    sample_frame = -1
    all_ft_frames = set()
    for track in structured_tracklets:
        for f in track['frames']:
            all_ft_frames.add(int(f))
            
    # 遍历 Graph 中的帧，找第一个交集
    for f in sorted(cells_by_t.keys()):
        if f in all_ft_frames:
            sample_frame = f
            break
            
    if sample_frame == -1:
        print(f"[ERROR] 视频 {video_id} 的轨迹数据与 Graph 数据在时间轴(Frame)上无交集！")
        return

    # 2. 提取该样本帧的坐标数据
    ft_boxes_at_frame = []
    for track in structured_tracklets:
        if sample_frame in track['frames']:
            # 找到对应帧的索引
            idx = track['frames'].index(sample_frame)
            ft_boxes_at_frame.append(track['bboxes'][idx]) # [x1, y1, x2, y2]

    # 提取 Graph Cell 数据 (cx, cy, w, h)
    cell_nodes_at_frame = [cell[1] for cell in cells_by_t[sample_frame]]

    print(f"[DEBUG] 样本帧: {sample_frame}")
    print(f"[DEBUG] FT 框数量: {len(ft_boxes_at_frame)}, Cell 节点数量: {len(cell_nodes_at_frame)}")

    # 3. 执行偏移诊断
    debugger = AlignmentDebugger(scale_factor=scale_factor)
    debugger.print_diagnostic_report(ft_boxes_at_frame, cell_nodes_at_frame)

def plot_paper_comparison(seq_name, baseline_df, optimized_df, gt_df, save_dir):
    
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 兼容性样式设置 ---
    available_styles = plt.style.available
    if 'seaborn-v0_8-paper' in available_styles:
        plt.style.use('seaborn-v0_8-paper')
    elif 'seaborn-paper' in available_styles:
        plt.style.use('seaborn-paper')
    else:
        plt.style.use('ggplot') # 最后的保底方案
        
    # 设置全局字体以适应论文（可选）
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 16
    })
    # ---------------------

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300)
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=300) # 300DPI 满足投稿要求
    # titles = ["(a) Baseline (Online Tracking)", "(b) Ours (Global Optimized)", "(c) Ground Truth"]
    dfs = [baseline_df, optimized_df, gt_df]
    
    # 统一坐标范围
    all_x = pd.concat([df['x1'] for df in dfs])
    all_y = pd.concat([df['y1'] for df in dfs])
    x_lim = (all_x.min() - 50, all_x.max() + 50)
    y_lim = (all_y.max() + 50, all_y.min() - 50)

    for i, df in enumerate(dfs):
        ax = axes[i]
        df['cx'] = (df['x1'] + df['x2']) / 2
        df['cy'] = (df['y1'] + df['y2']) / 2
        
        unique_ids = df['id'].unique()
        for tid in unique_ids:
            track = df[df['id'] == tid].sort_values('frame')
            # 随机但固定的颜色
            color = plt.cm.tab20(int(tid) % 20)
            ax.plot(track['cx'], track['cy'], color=color, linewidth=1.5, alpha=0.7)
            
            # 在轨迹起点画点，终点画箭头
            ax.scatter(track['cx'].iloc[0], track['cy'].iloc[0], color=color, s=10)
            if len(track) > 2:
                ax.annotate('', xy=(track['cx'].iloc[-1], track['cy'].iloc[-1]), 
                            xytext=(track['cx'].iloc[-2], track['cy'].iloc[-2]),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        # ax.set_title(titles[i], fontsize=20, pad=10)
        ax.set_xlim(x_lim); ax.set_ylim(y_lim)
        ax.axis('off') # 论文图通常去掉坐标轴更美观

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{seq_name}_comparison.pdf")) # 保存PDF矢量图，投稿必备
    # plt.show()

def save_all_frames_comparison(seq_name, baseline_df, optimized_df, img_dir, save_root):
    """
    全量逐帧对比保存工具 (纯 OpenCV 实现，速度快)。
    将每一帧的 Baseline (上) 和 Ours (下) 拼接成一张图并独立保存。
    """
    import os
    import cv2
    import numpy as np
    from tqdm import tqdm  # 用于显示进度条

    # 为当前视频创建一个独立的文件夹存放所有帧
    seq_save_dir = os.path.join(save_root, f"{seq_name}_all_frames")
    os.makedirs(seq_save_dir, exist_ok=True)
    
    print(f"[Viz] Rendering all frames for {seq_name} into {seq_save_dir}...")

    # 预计算中心点
    for df in [baseline_df, optimized_df]:
        if 'cx' not in df.columns:
            df['cx'] = (df['x1'] + df['x2']) / 2
            df['cy'] = (df['y1'] + df['y2']) / 2

    # 获取所有需要处理的帧（取 Baseline 和 Optimized 的并集）
    all_frames = sorted(list(set(baseline_df['frame']).union(set(optimized_df['frame']))))
    
    # 提取按 ID 分组的轨迹字典，大幅提高每一帧查询历史轨迹的速度
    base_tracks = {tid: grp.sort_values('frame') for tid, grp in baseline_df.groupby('id')}
    opt_tracks = {oid: grp.sort_values('frame') for oid, grp in optimized_df.groupby('id')}

    # 遍历所有帧并保存
    for frame_id in tqdm(all_frames, desc="Rendering Frames"):
        # 1. 读取图像
        img_path = os.path.join(img_dir, f"{int(frame_id):06d}.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, f"{int(frame_id)}.jpg")
            
        img_raw = cv2.imread(img_path)
        if img_raw is None:
            continue
            
        img_base = img_raw.copy()
        img_ours = img_raw.copy()

        # 2. 绘制 Baseline
        for tid, hist in base_tracks.items():
            # 获取当前帧及之前的轨迹
            hist_till_now = hist[hist['frame'] <= frame_id]
            if hist_till_now.empty: continue
            
            # 画历史轨迹线
            if len(hist_till_now) > 1:
                pts = np.array([hist_till_now['cx'].values, hist_till_now['cy'].values]).T.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(img_base, [pts], False, (0, 0, 255), 3) # 红色轨迹
                
            # 画当前帧 BBox
            if frame_id in hist_till_now['frame'].values:
                d = hist_till_now[hist_till_now['frame'] == frame_id].iloc[0]
                cv2.rectangle(img_base, (int(d['x1']), int(d['y1'])), (int(d['x2']), int(d['y2'])), (0, 0, 255), 4)

        # 3. 绘制 Ours (Optimized)
        for oid, hist in opt_tracks.items():
            hist_till_now = hist[hist['frame'] <= frame_id]
            if hist_till_now.empty: continue
            
            # 画历史轨迹线
            if len(hist_till_now) > 1:
                pts = np.array([hist_till_now['cx'].values, hist_till_now['cy'].values]).T.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(img_ours, [pts], False, (0, 255, 0), 3) # 绿色轨迹
                
            # 画当前帧 BBox
            if frame_id in hist_till_now['frame'].values:
                d = hist_till_now[hist_till_now['frame'] == frame_id].iloc[0]
                cv2.rectangle(img_ours, (int(d['x1']), int(d['y1'])), (int(d['x2']), int(d['y2'])), (0, 255, 0), 4)

        # 4. 添加文字标签 (左上角)
        # cv2.putText(img_base, f"Baseline - Frame: {int(frame_id)}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        # cv2.putText(img_ours, f"Ours (Optimized) - Frame: {int(frame_id)}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

        # 5. 上下拼接并保存 (Top: Baseline, Bottom: Ours)
        combined_img = np.vstack((img_base, img_ours))
        
        save_path = os.path.join(seq_save_dir, f"frame_{int(frame_id):06d}.jpg")
        cv2.imwrite(save_path, combined_img)



def testfast(opt, split, modelPath, show_flag, results_name, save_mat=False, i_th=3):
    total_frames = 0
    total_elapsed = 0.0
    timer = {
    'optical_flow': 0.0,   # preprocess 中的光流
    'backbone': 0.0,       # process() 推理
    'postprocess': 0.0,    # post_process + feature extract
    'mot_update': 0.0,     # mot_tracker.update()
    'global_opt': 0.0,     # optimize_and_interpolate_tracks
}
    
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    print("Using device:", opt.device)
    opt.test_large_size = True

    print(opt.model_name)

    dataset = get_dataset(opt)

    DataVal = dataset(opt, split)
    if opt.off_flag:
        head = {'hm': DataVal.num_classes, 'wh': 2, 'reg': 2}
    else:
        head = {'hm': DataVal.num_classes, 'wh': 2}
    print("data resolution:", DataVal.resolution)    
    model = get_det_net(head, opt.model_name, DataVal.resolution, opt.seqLen, opt, thresh=i_th)  # 建立模型
    
    model = load_model(model, modelPath)
    model = model.to(opt.device)
    model.eval()

    if hasattr(opt, "auto_test") and opt.auto_test:
    # 使用调参脚本传进来的目录
        track_results_save_dir = opt.track_results_dir
    else:
        # 正常测试才用新的 run_id
        # print("results save_results_dir:", opt.save_results_dir)
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        track_results_save_dir = os.path.join(opt.save_results_dir, f"fasttrackingResults_{opt.model_name}_{run_id}")
    print("Results will be saved to:", track_results_save_dir)
    os.makedirs(track_results_save_dir, exist_ok=True)
    track_result_file = os.path.join(track_results_save_dir, f"{results_name}.txt")
    track_file = open(track_result_file, 'w')

    # log_file = os.path.join(track_results_save_dir, "log.txt")

    return_time = False
    num_classes = dataset.num_classes
    max_per_image = opt.K

    if save_mat:
        save_mat_path_upper = os.path.join(opt.save_results_dir, results_name)
        if not os.path.exists(save_mat_path_upper):
            os.mkdir(save_mat_path_upper)

    test_upper_path = opt.data_dir + 'test/'
    print("Testing on data path:", test_upper_path)

    data_folder_list = os.listdir(test_upper_path)
    patch_len = opt.seqLen

    offline_det_root = "/home/liangcx/workspace/hieum-fast/data/output/no_road_txt1/"
    # 结果保存目录
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    track_results_save_dir = os.path.join(opt.save_results_dir, f"tracking_{run_id}")
    os.makedirs(track_results_save_dir, exist_ok=True)

    print(f"Reading offline detections from: {offline_det_root}")

    time_all = []
    det_result = {}
    all_video_tracklets = {}
    all_video_raw_dets = {}
    all_video_appearance_maps = {}

    global_analyzer = GlobalTrajectoryAnalyzer(iou_threshold=0.3)
    tuner = TrackSubsegmentTuner()

    cache_root = "./offline_cache"
    use_offline_cache = getattr(opt, "use_offline_cache", False)

    for ii in range(len(data_folder_list)):
        
        appearance_map = {} # 当前视频的特征图
        
        data_folder_path = os.path.join(test_upper_path, data_folder_list[ii], 'img1')
        print("Processing folder:", data_folder_path)
        sequence_result_file = os.path.join(track_results_save_dir, f"{data_folder_list[ii]}.txt")
        sequence_file = open(sequence_result_file, 'w')
        mot_tracker = Fasttracker(opt)
        if save_mat:
            save_mat_folder = os.path.join(save_mat_path_upper, data_folder_list[ii])
            if not os.path.exists(save_mat_folder):
                os.mkdir(save_mat_folder)
        img_list = os.listdir(data_folder_path)
        img_list = [i for i in img_list if i.endswith('.jpg')]
        img_list.sort()
        imgs_number = len(img_list)
        overlap_flag = 0
        if len(img_list)%patch_len==0:
            patch_num = len(img_list)//patch_len
        else:
            patch_num = len(img_list) // patch_len+1
            overlap_flag=1


        video_id = data_folder_list[ii]  # 当前视频名称
        img_list = sorted([i for i in os.listdir(data_folder_path) if i.endswith('.jpg')])
        img_len = len(img_list)
        frame_step = 1


        # 加载当前视频对应的离线检测文件
        # offline_txt_path = os.path.join(offline_det_root, f"{video_id}.txt")
        # offline_data = load_offline_dets(offline_txt_path)
        # 针对 Baseline 模式的结果容器
        # --- 尝试加载离线数据 ---
        cached_data = load_offline_cache(video_id, cache_root) if use_offline_cache else None
        
        
        # --- 如果没有缓存，则运行原始推理逻辑 ---
        current_video_det_results = {} # 暂存当前视频的检测
        baseline_video_data = []      # 暂存当前视频的跟踪
        # 构建 image_match_dict
        
        road_mask_path = f"/home/liangcx/datasets/crop_datasets/road_txt/test_{video_id}_road_mask.txt"
        road_mask = np.loadtxt(road_mask_path)
        if cached_data:
            print(f"Loading cached data for {video_id}...")
            current_video_det_results = cached_data['det_results']
            baseline_video_data = cached_data['tracklets']
            # 合并到全局供后续 Phase 使用
            det_result.update(current_video_det_results)
            all_video_tracklets[video_id] = baseline_video_data
            # 从缓存中恢复 raw_dets
            all_video_raw_dets[video_id] = cached_data.get('raw_dets', {})
            # 注意：这里不再 continue，而是让程序流向下面的“统一保存”逻辑
        else:
            # --- 如果没有缓存，运行原始推理逻辑 ---
            print(f"Running inference for {video_id}...")
        
            for pk in range(patch_num):
                time_start = time.time()
                if overlap_flag and pk==patch_num-1:
                    patch_ims = img_list[imgs_number-patch_len:imgs_number]
                else:
                    patch_ims = img_list[pk*patch_len : (pk+1)*patch_len]
                patch_ims_path = [os.path.join(data_folder_path, i) for i in patch_ims]
                batch_dict, meta, patch_imgs, input_imgs = preprocess(patch_ims_path, DataVal)
                for k in input_imgs:
                    if k == 'batch_size':
                        continue
                    input_imgs[k] = torch.from_numpy(input_imgs[k]).to(opt.device)
                # time_start1 = time.time()
                t_back = time.time()

                output,features, dets = process(model, input_imgs, return_time, opt, opt.K)
                torch.cuda.synchronize()
                # time_start3 = time.time()
                timer['backbone'] += time.time() - t_back

                # 后处理
                t_post = time.time()
                rets, dets_post = post_process(dets, meta, num_classes, max_per_image=max_per_image)
                timer['postprocess'] += time.time() - t_post

                video_id = data_folder_list[ii]
                
                for det_i in range(len(dets_post)):
                    frame_id = int(patch_ims[det_i].split('.')[0])  # 例如 '000005.jpg' -> 5
                    # dets_i = dets_post[det_i][1]  # 检测结果为 shape=(N, 5)，前四列为框，第五列为置信度
                    
                    # offline_dets = offline_data.get(frame_id, np.empty((0, 5)))

                    full_image_path = patch_ims_path[det_i]
                    frame_feature = features[:, :, det_i, :, :]
                    single_ret = rets[det_i]
                    ret = extract_features_for_class_1(single_ret,frame_feature)
                    
                    det_result[full_image_path] = ret
                    current_video_det_results[full_image_path] = ret 

                    if 1 in ret and 'f1' in ret:
                        for idx, feat in enumerate(ret['f1']):
                            # 这里使用 idx 和 frame_id 构造唯一 Key，必须与 build_global_graph 中的生成逻辑一致
                            det_key = f"det_{idx}_{frame_id}"
                            # 存入 CPU 内存，释放显存
                            appearance_map[det_key] = feat.detach().cpu().numpy()

                    if video_id not in all_video_raw_dets:
                        all_video_raw_dets[video_id] = {}
                        
                    dets_tracks = dets_post[det_i][1] # [N, 5]: [x1, y1, x2, y2, conf]
                    frame_dets_for_graph = []
                    
                    if dets_tracks is not None and len(dets_tracks) > 0:
                        for d in dets_tracks:
                            x1, y1, x2, y2 = float(d[0]), float(d[1]), float(d[2]), float(d[3])
                            conf = float(d[4]) if len(d) > 4 else 1.0
                            w = x2 - x1
                            h = y2 - y1
                            frame_dets_for_graph.append([x1, y1, w, h, conf])
                            
                    all_video_raw_dets[video_id][frame_id] = frame_dets_for_graph
                    # =========================================================

                    frame_id_abs = pk * patch_len + det_i + 1
                    dets_track = dets_post[det_i][1] # [x1, y1, x2, y2, conf]
                    t_assoc = time.time()
                    online_targets = mot_tracker.update(
                    # output_results=offline_dets,   # numpy array [N,5]
                    output_results=dets_track,   # numpy array [N,5]
                    img_info=(512,512),  # h, w
                    img_size=(512, 512),  # 模型输入分辨率，比如 (512,512)
                    road_mask=road_mask,
                    frame_id=pk * patch_len + frame_id + 1,
                    video_id=video_id,
                    det_result_all=det_result,
                    frame_step=frame_step,
                    img_len=img_len
                    )
                    timer['mot_update'] += time.time() - t_assoc

                    for t in online_targets:
                        t_id = t.track_id
                        tlwh = t.tlwh
                        score = t.score
                        baseline_video_data.append([
                            frame_id, 
                            t_id, 
                            tlwh[0], 
                            tlwh[1], 
                            tlwh[2], 
                            tlwh[3], 
                            score, 
                            -1, -1, -1
                        ])
                        # 将 tlwh 转回 x1, y1, x2, y2 方便匹配
                        t_box = [tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]]
                        
                        # 在当前帧的原始检测 ret[1] 中找最匹配的框
                        best_iou = 0
                        best_feat = None
                        if 1 in ret:
                            for idx, d_box in enumerate(ret[1]):
                                iou = compute_iou_x1y1x2y2(t_box, d_box[:4])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_feat = ret['f1'][idx]
                        
                        # 如果 IoU 足够高，说明这个特征就是该 Track 的
                        if best_iou > 0.7 and best_feat is not None:
                            appearance_map[(frame_id, t_id)] = best_feat.detach().cpu().numpy()
                    all_video_appearance_maps[video_id] = appearance_map
                time_end = time.time()
                patch_frames = len(patch_ims)
                total_frames += patch_frames
                total_elapsed += time_end - time_start
                print('time_used:', time_end - time_start, timer['backbone'], timer['postprocess'])
                # time_all.append(time_end - time_start1)
                gpus = GPUtil.getGPUs()
                gpu = gpus[0]
                print('patch_len: {} GPU used: {}/{}'.format(patch_len, gpu.memoryUsed, gpu.memoryTotal))
                
                ### view results
                # if save_mat:
                #     fig_save_name1 = os.path.join(save_mat_folder, '%03d_ori.png'%(pk+1))
                #     fig_save_name2 = os.path.join(save_mat_folder, '%03d_det.png' % (pk + 1))
                #     view_cloud(output['voxel_coords'], save_flag=1, fig_save_name = fig_save_name1)
                #     view_dets(dets, conf_th=0.3,save_flag=1, fig_save_name = fig_save_name2)
                if(show_flag):
                    hm1 = output['hm'].squeeze(0).squeeze(0).cpu().detach().numpy()
                    for det_i in range(len(dets_post)):
                        img = patch_imgs[:,:,:,det_i]
                        frame, _ = cv2_demo(img.astype(np.uint8), dets_post[det_i][1])

                        cv2.imshow('frame',frame)
                        cv2.waitKey(5)
                        hm2 = hm1[det_i]
                        cv2.imshow('hm', hm2)
                        cv2.waitKey(5)
                # if(show_flag):
                #     img = patch_imgs[:, :, :, det_i].copy() # 使用 copy 避免原地修改影响后续逻辑
                    
                #     # 【关键修改】：传入 offline_dets 而不是 dets_post[det_i][1]
                #     # 颜色改为 (255, 0, 0) 蓝色，方便区分离线和在线结果
                #     frame_vis, _ = cv2_demo(img.astype(np.uint8), offline_dets)

                #     cv2.imshow('Offline_Detection_Frame', frame_vis)
                #     cv2.waitKey(5)
                #     # 热力图展示（可选）
                #     hm1 = output['hm'].squeeze(0).squeeze(0).cpu().detach().numpy()
                #     hm2 = hm1[det_i]
                #     cv2.imshow('Heatmap', hm2)
                    
                #     cv2.waitKey(5)
                        

                # if save_mat:
                #     for ik in range(len(patch_ims)):
                #         mat_save_name = os.path.join(save_mat_folder, patch_ims[ik].replace('.jpg', '.mat'))
                #         ret = rets[ik]
                #         A = np.array(ret[1])
                #         scio.savemat(mat_save_name, {'A':A})
            # 如果当前不是从缓存读取的，则存入磁盘
            if not cached_data:
                save_offline_cache(video_id, current_video_det_results, baseline_video_data, all_video_raw_dets.get(video_id, {}), cache_root)
        # 视频处理结束后：
        if not getattr(opt, "use_global_optim", True):
            # Baseline 模式：直接保存该视频结果
            save_path = os.path.join(track_results_save_dir, f"{video_id}.txt")
            if baseline_video_data:
                res_np = np.array(baseline_video_data)
                res_np = res_np[res_np[:, 0].argsort()] # 按帧排序
                np.savetxt(save_path, res_np, fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d", delimiter=",")
            else:
                # 如果没检测到任何目标，也要创建一个空文件防止评估报错
                open(save_path, 'a').close()
            print(f"Results saved to {save_path}")
            all_video_tracklets[video_id] = baseline_video_data
            

        else:
            # Global 模式：存储 tracklets 供后续统一求解
            all_video_tracklets[video_id] = baseline_video_data
    
    best_mota = -1.0
    best_idf1 = -1.0
    best_config = None
    gt_dir = '/home/liangcx/datasets/crop_datasets/test/*/gt/gt.txt'
    if not getattr(opt, "use_global_optim", True):
        
        current_save_dir = os.path.join(opt.save_results_dir, f"{opt.model_name}")
        os.makedirs(current_save_dir, exist_ok=True)
        # # 设置保存目录
        
       #无全局优化
        best_conf = 0.3
        print(f"建议在 Phase 2 使用以下参数:")
        print("="*40 + "\n")

        if best_conf:
                print(f"\n>>> 正在验证推荐阈值: Conf >= {best_conf:.2f}")
                
                # 3. 应用阈值生成验证集
                # 创建临时验证目录
                val_save_dir = os.path.join(opt.save_results_dir, f"validation_conf_{best_conf:.2f}_{run_id}")
                os.makedirs(val_save_dir, exist_ok=True)
                
                for video_id, flat_tracklets in all_video_tracklets.items():
                    if not flat_tracklets: continue
                    
                    # --- 过滤逻辑 ---
                    # 策略：只保留置信度高于阈值的点（Point-wise Filtering）
                    # 这模拟了我们在 Phase 2 中提取 "High Confidence Seeds" 的过程
                    filtered_data = []
                    for item in flat_tracklets:
                        # item: [frame, id, x, y, w, h, conf, ...]
                        conf = item[6]
                        if conf >= best_conf:
                            # 转换为 MOT 格式保存
                            filtered_data.append([
                                item[0], item[1], 
                                item[2], item[3], item[4], item[5], 
                                conf, -1, -1, -1
                            ])
                    
                    # 保存验证文件
                    if filtered_data:
                        res_np = np.array(filtered_data)
                        res_np = res_np[res_np[:, 0].argsort()]
                        save_path = os.path.join(val_save_dir, f"{video_id}.txt")
                        np.savetxt(save_path, res_np, fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d", delimiter=",")

                # 4. 立即评估验证集
                print(f"正在评估验证集: {val_save_dir}")
                gt_dir = '/home/liangcx/datasets/crop_datasets/test/*/gt/gt.txt'
                val_result_dir = os.path.join(val_save_dir, "*.txt")
                
                print("-" * 30)
                print(f" [验证结果] 阈值 {best_conf:.2f}")
                run_evaluation(gt_dir, val_result_dir)
        else:
            current_save_dir = os.path.join(opt.save_results_dir, f"{opt.model_name}")
            os.makedirs(current_save_dir, exist_ok=True)
            # # 设置保存目录

            #全局优化
            for video_id, baseline_video_tracklets in all_video_tracklets.items():
                print(f"\nProcessing video {video_id} with Global Optimization...")
                current_video_raw_dets = all_video_raw_dets.get(video_id, {})
                current_appearance_map = all_video_appearance_maps.get(video_id, {})
                t_global = time.time()
                track_data = optimize_and_interpolate_tracks_1(baseline_video_tracklets, current_appearance_map)
                timer['global_opt'] += time.time() - t_global
                save_path = os.path.join(current_save_dir, f"{video_id}.txt")
                if len(track_data) == 0:
                    # 创建一个空的但带有列定义的 DataFrame，防止 eval 崩溃
                    with open(save_path, 'w') as f:
                        f.write("") # 或者写入 MOT 格式的 dummy line
                    print(f"Warning: Video {video_id} has no tracks.")
                print(f"Saving tracking results to {save_path}")
                np.savetxt(save_path, track_data, fmt="%.2f", delimiter=",")

            print("\n=== Finished: Fasttracker Only (Baseline) ===")
            gt_dir = '/home/liangcx/datasets/crop_datasets/test/*/gt/gt.txt' # 您的 GT 路径模式
            # result_dir = os.path.join(track_results_save_dir, "*.txt")
            result_dir = os.path.join(current_save_dir, "*.txt")
            print(f"result_dir: {result_dir}")
            summary = run_evaluation(gt_dir, result_dir, visualize=True) # 假设返回 DataFrame 或 Series
        

    # ------------------------------------------------------------------
    # 6. 输出最终最佳结果
    # ------------------------------------------------------------------
    print("\n========================================")
    print("Optimization Completed.")
    print(f"Best MOTA: {best_mota:.4f}")
    print(f"Best IDF1: {best_idf1:.4f}")
    print("Best Parameters:")
    # for k, v in best_config.items():
    #     print(f"  {k}: {v}")
    print("========================================")
    


    total = sum(timer.values())
    print("\n=== Timing Breakdown ===")
    for k, v in timer.items():
        print(f"  {k}: {v:.3f}s ({v/total*100:.1f}%)")
    print(f"  Total: {total:.3f}s")
    if total_elapsed > 0:
        fps = total_frames / total_elapsed
        print(f"\n=== Overall FPS: {fps:.2f} (over {total_frames} frames, {total_elapsed:.2f}s) ===")
    results_return = {}
    return results_return

def cv2_demo_fixed(frame, detections, confidence_threshold=0.3):
    """
    每一帧调用时，传入的是该帧的原始图像。
    函数会在该帧的副本或原始数据上绘制，绘制完成后通过 imshow 覆盖窗口内容。
    """
    # 确保不修改原始传入的图像引用（可选，视具体逻辑而定）
    # 如果 frame 已经是这一帧新读取的 jpg 数组，则不需要 copy
    canvas = frame.copy() 
    
    det_out = []
    if canvas.dtype != np.uint8:
        canvas = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    for i in range(detections.shape[0]):
        conf = detections[i, 4]
        if conf >= confidence_threshold:
            x1, y1, x2, y2 = detections[i, 0:4]
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            
            # 绘制当前帧的矩形
            cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), (0, 255, 0), 2)
            cv2.putText(canvas, f"{conf:.2f}", (ix1, max(0, iy1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            det_out.append([ix1, iy1, ix2, iy2, conf])
            
    return canvas, det_out

def visualize_tracking_results(save_dir, test_upper_path, tracking_results, output_video=False):
    """
    在原始图像上可视化跟踪结果。
    save_dir: 跟踪结果txt所在文件夹
    test_upper_path: 数据集 test/ 根目录
    tracking_results: Fasttracker 输出的 dict
    output_video: 是否保存为视频文件
    """
    for video_id, track_data in tracking_results.items():
        # --- 构建视频路径 ---
        seq_path = os.path.join(test_upper_path, video_id, "img1")
        img_list = sorted([f for f in os.listdir(seq_path) if f.endswith(".jpg")])
        if len(img_list) == 0:
            print(f"[Warning] {video_id} 没有找到图像，跳过。")
            continue

        # --- 随机为每个track_id分配颜色 ---
        track_colors = {}
        for tid in np.unique(track_data[:, 1]):
            track_colors[int(tid)] = tuple([int(x) for x in np.random.randint(0, 255, 3)])

        # --- 如果要保存视频 ---
        if output_video:
            first_img = cv2.imread(os.path.join(seq_path, img_list[0]))
            h, w = first_img.shape[:2]
            video_save_path = os.path.join(save_dir, f"{video_id}_vis.avi")
            out = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'XVID'), 20, (w, h))
            print(f"[Video] Saving visualization to {video_save_path}")

        # --- 按帧可视化 ---
        for img_name in img_list:
            frame_id = int(img_name.split(".")[0])
            img_path = os.path.join(seq_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 当前帧对应的轨迹
            frame_tracks = track_data[track_data[:, 0] == frame_id]
            for tr in frame_tracks:
                track_id = int(tr[1])
                x, y, w, h = tr[2:6]
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                color = track_colors.get(track_id, (0, 255, 0))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"ID:{track_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if output_video:
                out.write(img)
            else:
                cv2.imshow(f"Tracking - {video_id}", img)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        if output_video:
            out.release()
            print(f"[Done] Video saved: {video_save_path}")

    cv2.destroyAllWindows()