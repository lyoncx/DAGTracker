# eval_tracking.py
import os
import numpy as np
import pandas as pd
import motmetrics as mm
import glob
import cv2 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
# def read_data(file_path, is_gt=False):
#     """ Read GT or prediction data """
#     if is_gt:
#         return pd.read_csv(
#             file_path,
#             header=None,
#             names=['frame', 'id', 'left', 'top', 'width', 'height',
#                    'conf', 'x', 'y', 'z', 'a', 'b', 'c'],
#             index_col=False
#         )
#     else:
#         return pd.read_csv(
#             file_path,
#             header=None,
#             names=['frame', 'id', 'left', 'top', 'width', 'height'],
#             index_col=False
#         )

def read_data(file_path, is_gt=False):
    """ Read GT or prediction data """
    if is_gt:
        # GT 通常有 10-13 列
        return pd.read_csv(
            file_path,
            header=None,
            names=['frame', 'id', 'left', 'top', 'width', 'height',
                   'conf', 'x', 'y', 'z', 'a', 'b', 'c'][:13], # 截取实际存在的列
            index_col=False
        )
    else:
        # 重点：预测结果通常包含 'conf' 列，必须定义出来，否则 pandas 会报错或错位
        # 即使你后面不使用 conf，也要把它读进来
        try:
            # 尝试读取，不限制列数，先看文件到底有多少列
            df_temp = pd.read_csv(file_path, header=None, nrows=5)
            col_count = df_temp.shape[1]
            
            # 动态生成列名，防止 ParserWarning
            base_names = ['frame', 'id', 'left', 'top', 'width', 'height', 'conf', 'x', 'y', 'z']
            actual_names = base_names[:col_count] if col_count <= len(base_names) else base_names + [f'extra_{i}' for i in range(col_count - len(base_names))]
            
            return pd.read_csv(
                file_path,
                header=None,
                names=actual_names,
                index_col=False
            )
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
            return pd.DataFrame()

def convert_bbox_to_mot(df):
    """ Convert bounding box to MOT challenge format """
    df['x1'] = df['left']
    df['y1'] = df['top']
    df['x2'] = df['left'] + df['width']
    df['y2'] = df['top'] + df['height']
    cols = ['frame', 'id', 'x1', 'y1', 'x2', 'y2']
    if 'conf' in df.columns:
        cols.append('conf')
    else:
        df['conf'] = 1.0 # 默认置信度为1
        cols.append('conf')
    return df[cols]

def evaluate(truths, predictions):
    """ Evaluate tracking performance using MOT metrics """
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(set(truths['frame'].unique()).union(set(predictions['frame'].unique()))):
        gt = truths[truths['frame'] == frame]
        pr = predictions[predictions['frame'] == frame]
        gt_boxes = gt[['x1', 'y1', 'x2', 'y2']].values
        pr_boxes = pr[['x1', 'y1', 'x2', 'y2']].values
        distances = mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
        acc.update(
            gt['id'].values,
            pr['id'].values,
            distances
        )
    return acc

# ==========================================
# 新增：可视化功能模块
# ==========================================
def get_color(idx):
    """根据 ID 生成唯一的颜色"""
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def visualize_trajectory(seq_name, predictions, img_dir, save_root, ground_truths):
    """
    将 GT 和 Prediction 的轨迹结果分别绘制在原图上，并保存为图片序列。
    保存路径结构：
    save_root/seq_name/trajectory_gt/frame_id.jpg
    save_root/seq_name/trajectory_pred/frame_id.jpg
    """
    if not os.path.exists(img_dir):
        print(f"[Viz] 警告: 未找到图像目录: {img_dir}。跳过可视化。")
        return

    # 创建保存图片的目录
    gt_frames_dir = os.path.join(save_root, "trajectory_gt", seq_name)
    pred_frames_dir = os.path.join(save_root, "trajectory_noroad_pred", seq_name)
    os.makedirs(gt_frames_dir, exist_ok=True)
    os.makedirs(pred_frames_dir, exist_ok=True)
    
    img_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_list:
        print(f"[Viz] 在 {img_dir} 中未找到图像")
        return

    # 轨迹历史记录
    pred_trace_history = {}
    gt_trace_history = {}
    max_trace_len = 9999 # 轨迹尾巴的长度

    # 数据分组
    pred_grouped = predictions.groupby('frame')
    gt_grouped = ground_truths.groupby('frame')

    print(f"[Viz] 正在为 {seq_name} 生成逐帧轨迹对比图...")

    for img_path in img_list:
        # 提取文件名作为保存的文件名
        filename = os.path.basename(img_path)
        # 提取 frame_id 用于数据查找
        try:
            frame_id = int(os.path.splitext(filename)[0])
        except ValueError:
            # 如果文件名不是纯数字，尝试 MOT 的命名规则 (如 000001.jpg)
            continue

        img_raw = cv2.imread(img_path)
        if img_raw is None: continue
        
        # 克隆两份原始图像，一份画GT，一份画预测
        img_gt = img_raw.copy()
        img_pred = img_raw.copy()

        # --- 1. 绘制 Ground Truth (绿色风格) ---
        if frame_id in gt_grouped.groups:
            current_gts = gt_grouped.get_group(frame_id)
            for _, row in current_gts.iterrows():
                tid = int(row['id'])
                x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
                gt_color = (0, 255, 0) # BGR 绿色
                
                # 画 BBox
                cv2.rectangle(img_gt, (x1, y1), (x2, y2), gt_color, 2)
                # 画 ID 文本
                # cv2.putText(img_gt, f"GT:{tid}", (x1, max(y1-5, 10)), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 2)

                # 更新并绘制轨迹
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                if tid not in gt_trace_history: gt_trace_history[tid] = []
                gt_trace_history[tid].append(center)
                if len(gt_trace_history[tid]) > max_trace_len: gt_trace_history[tid].pop(0)
                
                if len(gt_trace_history[tid]) > 1:
                    pts = np.array(gt_trace_history[tid], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_gt, [pts], False, gt_color, 2)

        # --- 2. 绘制 Prediction (彩色风格) ---
        if frame_id in pred_grouped.groups:
            current_dets = pred_grouped.get_group(frame_id)
            for _, row in current_dets.iterrows():
                tid = int(row['id'])
                x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
                conf = row['conf'] if 'conf' in row else 1.0
                color = (255, 0, 0) # 预测结果不同ID使用不同颜色
                
                # 画 BBox
                cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
                # 画 ID 和置信度
                # label = f"ID:{tid} {conf:.2f}"
                # cv2.putText(img_pred, label, (x1, max(y1-5, 10)), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # 更新并绘制轨迹
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                if tid not in pred_trace_history: pred_trace_history[tid] = []
                pred_trace_history[tid].append(center)
                if len(pred_trace_history[tid]) > max_trace_len: pred_trace_history[tid].pop(0)

                if len(pred_trace_history[tid]) > 1:
                    pts = np.array(pred_trace_history[tid], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_pred, [pts], False, color, 2)

        # 添加左上角文字标识图片类型
        # cv2.putText(img_gt, f"SEQ: {seq_name} | Frame: {frame_id} | GROUND TRUTH", (20, 40), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # cv2.putText(img_pred, f"SEQ: {seq_name} | Frame: {frame_id} | PREDICTION", (20, 40), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 保存图片到对应的文件夹
        cv2.imwrite(os.path.join(gt_frames_dir, filename), img_gt)
        cv2.imwrite(os.path.join(pred_frames_dir, filename), img_pred)

    print(f"[Viz] 逐帧轨迹图已保存至: \n  - {gt_frames_dir}\n  - {pred_frames_dir}")

def plot_enhanced_comparison(seq_name, pred_df, gt_df, save_root):
    """
    生成高清晰度轨迹对比图，解决了样式兼容性问题并增强了肉眼辨识度
    """
    save_dir = os.path.join(save_root, seq_name)
    os.makedirs(save_dir, exist_ok=True)

    # 1. 样式兼容性处理
    available_styles = plt.style.available
    if 'seaborn-v0_8-whitegrid' in available_styles:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    else:
        plt.style.use('ggplot') # 备选常用样式

    # 2. 预处理数据
    for df in [pred_df, gt_df]:
        if 'cx' not in df.columns:
            df['cx'] = (df['x1'] + df['x2']) / 2
            df['cy'] = (df['y1'] + df['y2']) / 2

    # 3. 创建画布
    fig = plt.figure(figsize=(22, 14), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.25, wspace=0.15)
    
    # 获取全局坐标范围（用于对齐左右图）
    all_x = pd.concat([pred_df['cx'], gt_df['cx']])
    all_y = pd.concat([pred_df['cy'], gt_df['cy']])
    x_lim = (all_x.min() - 30, all_x.max() + 30)
    y_lim = (all_y.max() + 30, all_y.min() - 30) # 图像坐标Y轴反转

    # --- 1.1 左上：GT 轨迹 (强调路径连续性) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title(f"Ground Truth Tracks - {seq_name}", fontsize=18, fontweight='bold', pad=15)
    gt_ids = gt_df['id'].unique()
    # 使用高对比度颜色循环
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    
    for i, tid in enumerate(gt_ids):
        track = gt_df[gt_df['id'] == tid].sort_values('frame')
        color = colors[i % len(colors)]
        ax1.plot(track['cx'], track['cy'], color=color, linewidth=2, alpha=0.8, label=f'ID:{tid}' if len(gt_ids)<10 else "")
        # 画个箭头表示方向
        if len(track) > 5:
            mid = len(track) // 2
            ax1.annotate('', xy=(track['cx'].iloc[mid+1], track['cy'].iloc[mid+1]), 
                         xytext=(track['cx'].iloc[mid], track['cy'].iloc[mid]),
                         arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # --- 1.2 右上：Pred 轨迹 (置信度配色) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("Predicted Tracks (Color by Confidence)", fontsize=18, fontweight='bold', pad=15)
    pred_ids = pred_df['id'].unique()
    
    norm = Normalize(vmin=0.4, vmax=1.0) 
    cmap = plt.cm.plasma # 紫色(低) -> 黄色(高)
    
    for tid in pred_ids:
        track = pred_df[pred_df['id'] == tid].sort_values('frame')
        if len(track) < 2: continue
        points = np.array([track['cx'].values, track['cy'].values]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, array=track['conf'].values, linewidths=2.5)
        ax2.add_collection(lc)

    # 统一坐标轴
    for ax in [ax1, ax2]:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_xlabel("X Pixel", fontsize=12)
        ax.set_ylabel("Y Pixel", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)

    # 添加置信度图例
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar_ax = fig.add_axes([0.92, 0.6, 0.015, 0.25])
    fig.colorbar(sm, cax=cbar_ax, label='Confidence')

    # --- 2.1 左下：生命周期甘特图 (对比 ID 切换和断裂) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("Timeline Comparison (GT vs Pred)", fontsize=16)
    
    # 绘制 GT 生命周期
    for i, tid in enumerate(gt_ids):
        t = gt_df[gt_df['id'] == tid]
        ax3.plot([t['frame'].min(), t['frame'].max()], [i, i], color='green', linewidth=4, alpha=0.3)
    
    # 绘制 Pred 生命周期 (带偏移)
    offset = len(gt_ids) + 2
    for i, tid in enumerate(pred_ids):
        t = pred_df[pred_df['id'] == tid]
        ax3.plot([t['frame'].min(), t['frame'].max()], [i + offset, i + offset], color='blue', linewidth=2, alpha=0.6)
    
    ax3.axhline(y=offset-1, color='black', linestyle='-', alpha=0.2)
    ax3.set_xlabel("Frame Number")
    ax3.set_ylabel("Object Index (Bottom: GT | Top: Pred)")

    # --- 2.2 右下：关键指标文本框 (直接显示结果) ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    # 简单的统计信息
    stats_text = (
        f"Sequence: {seq_name}\n\n"
        f"GT Total IDs: {len(gt_ids)}\n"
        f"Pred Total IDs: {len(pred_ids)}\n"
        f"Avg Confidence: {pred_df['conf'].mean():.2f}\n"
        f"Frame Range: {int(all_x.index.min())} - {int(all_x.index.max())}\n\n"
        "Visual Guide:\n"
        "• Left-Top: Different colors = Different GT IDs\n"
        "• Right-Top: Yellow = High Conf, Purple = Low Conf\n"
        "• Bottom-Left: Gap in lines = Tracking Lost"
    )
    ax4.text(0.1, 0.5, stats_text, fontsize=14, family='monospace', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=1'))

    # 保存
    final_path = os.path.join(save_dir, f"{seq_name}_clear_comparison.png")
    plt.savefig(final_path, bbox_inches='tight')
    plt.close()
    print(f"✅ 增强对比图已成功保存至: {final_path}")

def visualize_frames(seq_name, predictions, img_dir, save_root, ground_truths):
    """
    将 GT 和 Prediction 结果分别保存为单帧图片
    """
    if not os.path.exists(img_dir):
        print(f"[Viz] 错误: 未找到图像目录 {img_dir}")
        return

    # 创建保存目录：save_root/seq_name/gt_frames 和 save_root/seq_name/pred_frames
    gt_save_dir = os.path.join(save_root, seq_name, "gt_frames")
    pred_save_dir = os.path.join(save_root, seq_name, "pred_frames")
    os.makedirs(gt_save_dir, exist_ok=True)
    os.makedirs(pred_save_dir, exist_ok=True)
    
    img_list = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not img_list:
        print(f"[Viz] 在 {img_dir} 中未找到图片")
        return

    # 轨迹历史记录 (用于绘制拖尾效果)
    pred_trace_history = {}
    gt_trace_history = {}
    max_trace_len = 50 

    # 按帧分组
    pred_grouped = predictions.groupby('frame')
    gt_grouped = ground_truths.groupby('frame')

    print(f"[Viz] 正在保存 {seq_name} 的逐帧对比图...")

    for img_path in img_list:
        filename = os.path.basename(img_path)
        # 尝试从文件名提取 frame_id，如果文件名不是纯数字则需特殊处理
        try:
            frame_id = int(os.path.splitext(filename)[0])
        except ValueError:
            continue

        img_raw = cv2.imread(img_path)
        if img_raw is None: continue
        
        img_gt = img_raw.copy()
        img_pred = img_raw.copy()

        # --- 1. 绘制 Ground Truth ---
        if frame_id in gt_grouped.groups:
            current_gts = gt_grouped.get_group(frame_id)
            for _, row in current_gts.iterrows():
                tid = int(row['id'])
                x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
                color = (0, 255, 0) # GT 统一用绿色
                
                cv2.rectangle(img_gt, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_gt, f"GT:{tid}", (x1, max(y1-5, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 轨迹绘制
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                if tid not in gt_trace_history: gt_trace_history[tid] = []
                gt_trace_history[tid].append(center)
                if len(gt_trace_history[tid]) > max_trace_len: gt_trace_history[tid].pop(0)
                if len(gt_trace_history[tid]) > 1:
                    pts = np.array(gt_trace_history[tid], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_gt, [pts], False, color, 2)

        # --- 2. 绘制 Prediction ---
        if frame_id in pred_grouped.groups:
            current_dets = pred_grouped.get_group(frame_id)
            for _, row in current_dets.iterrows():
                tid = int(row['id'])
                x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
                conf = row['conf'] if 'conf' in row else 1.0
                color = get_color(tid) # 预测用 ID 区分颜色
                
                cv2.rectangle(img_pred, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_pred, f"ID:{tid} {conf:.2f}", (x1, max(y1-5, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 轨迹绘制
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                if tid not in pred_trace_history: pred_trace_history[tid] = []
                pred_trace_history[tid].append(center)
                if len(pred_trace_history[tid]) > max_trace_len: pred_trace_history[tid].pop(0)
                if len(pred_trace_history[tid]) > 1:
                    pts = np.array(pred_trace_history[tid], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_pred, [pts], False, color, 2)

        # 保存图片
        cv2.imwrite(os.path.join(gt_save_dir, filename), img_gt)
        cv2.imwrite(os.path.join(pred_save_dir, filename), img_pred)

    print(f"[Viz] {seq_name} 处理完成。图片保存在: {os.path.join(save_root, seq_name)}")

def run_evaluation(gt_dir_pattern, result_dir_pattern, visualize=False, vis_save_root=None):
    """ Run evaluation for all GT and result file pairs """
    all_summaries = []
    for truth_path, result_path in zip(sorted(glob.glob(gt_dir_pattern)),
                                       sorted(glob.glob(result_dir_pattern))):
        
        seq_name = result_path.split('/')[-1].split('.')[0] 
        print(f'\nProcessing {seq_name}...')

        truths = read_data(truth_path, is_gt=True)
        predictions = read_data(result_path, is_gt=False)

        # Convert bbox coordinates
        truths = convert_bbox_to_mot(truths)
        predictions = convert_bbox_to_mot(predictions)

        # Evaluate tracking
        acc = evaluate(truths, predictions)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=[
            'mota', 'idf1', 'mostly_tracked', 'mostly_lost',
            'num_false_positives', 'num_misses', 'num_switches'
        ], name='metrics')
        all_summaries.append(summary)

        print(f'Results for {result_path}:')
        print(summary)

        # 3. 可视化 (如果开启)
        if visualize:
            # 推断图片目录
            base_dir = os.path.dirname(os.path.dirname(truth_path))
            img_dir = os.path.join(base_dir, 'img1')
            
            # if vis_save_root is None:
            #     vis_save_root = os.path.join(os.path.dirname(result_path), "viz_videos")
            if vis_save_root is None:
                vis_save_root = "vis_output"
            # 关键修改：传入 truths 进行对比可视化
            # visualize_frames(seq_name, predictions, img_dir, vis_save_root, ground_truths=truths)
            visualize_trajectory(seq_name, predictions, img_dir, vis_save_root, ground_truths=truths)

            # B. 【新增】生成轨迹分析图谱
            # 这不需要读取图片，速度很快，建议务必保留
            # plot_enhanced_comparison(seq_name, predictions, truths, vis_save_root)

    # Aggregate all summaries
    overall_summary = pd.concat(all_summaries).mean(axis=0)
    print("Overall Performance:")
    print(overall_summary)
    return overall_summary