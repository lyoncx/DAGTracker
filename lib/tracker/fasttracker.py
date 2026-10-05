import numpy as np
from collections import defaultdict, deque
import os
import os.path as osp
import copy
import torch.nn as nn
import torch
import torch.nn.functional as F
import json

from lib.models.module import Cnn_Module,AEloss
from .kalman_filter import KalmanFilter
from lib.tracker import matching
from .basetrack import BaseTrack, TrackState



class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    count = 0  # 自增 ID

    def __init__(self, tlwh, score, class_id=0):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.not_matched = 0
        self.is_occluded = False
        self.occluded_len = 0
        self.last_occluded_frame = -1
        self.was_recently_occluded = False
        self.mean_history = []

        self.track_id = STrack.count
        STrack.count += 1


    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)


    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov



    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.mean_history.append(self.mean.copy())
        if len(self.mean_history) > 100:  # limit history length
            self.mean_history.pop(0)


        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        self.history[frame_id] = (
            self.tlwh, 
            self.score, 
            getattr(self, 'features', None), # 假设新轨迹在初始化时就有特征
            self.mean[4], 
            self.mean[5]
        )


    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.mean_history.append(self.mean.copy())
        if len(self.mean_history) > 100:  # limit history length
            self.mean_history.pop(0)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        
        # ✅ 记录重新匹配的帧数据
        self.history[frame_id] = (
            self.tlwh, 
            self.score, 
            getattr(new_track, 'features', None), 
            self.mean[4], 
            self.mean[5]
        )

    def update_from_tlbr(self):
        """Recompute tlwh and mean from current tlbr"""
        x1, y1, x2, y2 = self.tlbr
        self._tlwh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
        if hasattr(self, 'mean'):
            self.mean[:4] = self.tlwh_to_xyah(self.tlwh)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        
        self.mean_history.append(self.mean.copy())
        if len(self.mean_history) > 100:  # limit history length
            self.mean_history.pop(0)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

        vx = self.mean[4] 
        vy = self.mean[5] 
        
        # 假设没有外观特征，使用 None 或一个占位符
        appearance_feature = getattr(new_track, 'features', None) 

        # 记录到 history 中
        self.history[frame_id] = (
            self.tlwh, 
            self.score, 
            appearance_feature, 
            vx, 
            vy
        )
        # 控制历史记录长度，可选：
        if len(self.history) > self.max_history_len:
            min_frame_id = min(self.history.keys())
            del self.history[min_frame_id]

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


def is_occluded_by(box_a, box_b, iou_thresh=0.7):
    """Returns True if box_a is significantly overlapped by box_b"""
    inter = (
        max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])) *
        max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]))
    )
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    if area_a == 0:
        return False
    iou = inter / area_a
    return iou > iou_thresh

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)



class Fasttracker(object):

    def __init__(self, opt, frame_rate=20):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.opt = opt

        self.det_thresh = opt.track_thresh
        self.match_thresh = opt.match_thresh
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size

        self.reset_velocity_offset_occ = opt.reset_velocity_offset_occ
        self.reset_pos_offset_occ = opt.reset_pos_offset_occ
        self.enlarge_bbox_occ = opt.enlarge_bbox_occ
        self.dampen_motion_occ = opt.dampen_motion_occ
        self.active_occ_to_lost_thresh = opt.active_occ_to_lost_thresh
        self.init_iou_suppress = opt.init_iou_suppress

        self.det_result_all = None
        self.next_id = 0
        self.kalman_filter = KalmanFilter()


        #道路约束
        self.use_soft_decay = False
        self.use_hard_road_filter = False
        self.use_topology_lifecycle = False
        

        self.gamma = 0.75          # 软衰减系数
        self.edge_margin = 95      # 画面边缘阈值
        self.soft_low_thr = 0.55  # 低分框下界
        self.soft_high_thr = self.det_thresh  # 高分框上界
        self._road_dist_cache = None
        self._road_dist_cache_shape = None

    def _is_on_road(self, tlbr, road_mask):
        """
        判断检测框的参考点是否在 road_mask 定义的合法区域内
        tlbr: [x1, y1, x2, y2]
        """
        if road_mask is None:
            return True
        
        # 取底部中心点作为参考点（最符合地面目标特征）
        cx = int((tlbr[0] + tlbr[2]) / 2)
        cy = int(tlbr[3]) # 底部边缘
        
        # 边界检查
        if 0 <= cy < road_mask.shape[0] and 0 <= cx < road_mask.shape[1]:
            return road_mask[cy, cx] > 0 # 假设 1 或 255 为道路
        return False

    def _get_road_dist_map(self, road_mask):
        import cv2

        if road_mask is None:
            return None

        mask = road_mask
        if mask.ndim == 3:
            mask = mask[..., 0]

        mask = (mask > 0).astype(np.uint8)
        cache_shape = mask.shape

        if self._road_dist_cache is not None and self._road_dist_cache_shape == cache_shape:
            return self._road_dist_cache

        bg_mask = (mask == 0).astype(np.uint8) * 255
        self._road_dist_cache = cv2.distanceTransform(bg_mask, cv2.DIST_L2, 3)
        self._road_dist_cache_shape = cache_shape
        return self._road_dist_cache

    def update(self, output_results, img_info, img_size, road_mask=None, frame_id=None, video_id=None,
            det_result_all=None, frame_step=None, img_len=None):

        if det_result_all is not None:
            self.det_result_all = det_result_all

        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # ------------------------------------------------------------
        # 1) 解析检测结果
        # ------------------------------------------------------------
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4].copy()   # x1, y1, x2, y2
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4].copy()   # x1, y1, x2, y2

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale  # 统一到原图坐标系

        # 原始高分检测掩码
        remain_inds = scores > self.det_thresh

        # ------------------------------------------------------------
        # 2) 软衰减：只改分数，不做硬剔除
        # ------------------------------------------------------------
        if road_mask is not None and self.use_soft_decay:
            dist_map = self._get_road_dist_map(road_mask)

            calibrated_scores = np.copy(scores)
            for i, box in enumerate(bboxes):
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)

                cx = np.clip(cx, 0, road_mask.shape[1] - 1)
                cy = np.clip(cy, 0, road_mask.shape[0] - 1)

                dist_to_road = dist_map[cy, cx]
                if dist_to_road > 0:
                    decay_factor = np.exp(-self.gamma * dist_to_road)
                    calibrated_scores[i] *= decay_factor

            scores = calibrated_scores
            remain_inds = scores > self.det_thresh

        # ------------------------------------------------------------
        # 3) 硬道路过滤：只在开启时进行
        #    这里既影响高分框保留，也影响新ID初始化
        # ------------------------------------------------------------
        if road_mask is not None and self.use_hard_road_filter:
            road_keep = np.zeros(len(bboxes), dtype=bool)
            for i, box in enumerate(bboxes):
                if remain_inds[i]:
                    road_keep[i] = self._is_on_road(box, road_mask)
            remain_inds = np.logical_and(remain_inds, road_keep)

        # 低分框二次关联
        inds_low = scores > self.soft_low_thr
        inds_high = scores < self.soft_high_thr
        inds_second = np.logical_and(inds_low, inds_high)

        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        dets_second = bboxes[inds_second]
        scores_second = scores[inds_second]

        if len(dets) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s)
                        for (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        # ------------------------------------------------------------
        # 4) 拆分当前轨迹
        # ------------------------------------------------------------
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # ------------------------------------------------------------
        # 5) 第一阶段关联
        # ------------------------------------------------------------
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)

        dists = matching.iou_distance(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
            track.is_occluded = False
            track.not_matched = 0
            track.occluded_len = 0

        # ------------------------------------------------------------
        # 6) 第二阶段：低分框关联
        # ------------------------------------------------------------
        if len(dets_second) > 0:
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s)
                                for (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

            track.is_occluded = False
            track.not_matched = 0
            track.occluded_len = 0

        # ------------------------------------------------------------
        # 7) 遮挡处理
        # ------------------------------------------------------------
        for it in u_track:
            track = r_tracked_stracks[it]
            track.not_matched += 1

            if not track.is_occluded and track.state == TrackState.Tracked:
                for other in activated_stracks:
                    if track.track_id == other.track_id:
                        continue
                    if not other.is_activated or other.is_occluded:
                        continue

                    if is_occluded_by(track.tlbr, other.tlbr):
                        track.is_occluded = True
                        track.occluded_len += 1
                        track.last_occluded_frame = self.frame_id
                        track.was_recently_occluded = True

                        if len(track.mean_history) >= self.reset_velocity_offset_occ:
                            old_mean = track.mean_history[-self.reset_velocity_offset_occ]
                            track.mean[4:8] = old_mean[4:8]

                        if len(track.mean_history) >= self.reset_pos_offset_occ:
                            old_mean = track.mean_history[-self.reset_pos_offset_occ]
                            track.mean[0:4] = old_mean[0:4]

                        if track.occluded_len == 1:
                            track.mean[3] *= self.enlarge_bbox_occ

                        track.mean[4:8] *= self.dampen_motion_occ
                        break

            if not track.is_occluded:
                track.occluded_len = 0
            else:
                track.occluded_len += 1

            if track.was_recently_occluded and (self.frame_id - track.last_occluded_frame > 40):
                track.was_recently_occluded = False

            if track.state != TrackState.Lost:
                if track.not_matched > 2 and (
                    not track.is_occluded or track.occluded_len > self.active_occ_to_lost_thresh
                ):
                    track.mark_lost()
                    lost_stracks.append(track)

        # ------------------------------------------------------------
        # 8) 未确认轨迹处理
        # ------------------------------------------------------------
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_lost()
            lost_stracks.append(track)

        # ------------------------------------------------------------
        # 9) 新轨迹初始化：只有在硬道路开关打开时才强制要求在路上
        # ------------------------------------------------------------
        active_now = {t.track_id: t for t in self.tracked_stracks if t.state == TrackState.Tracked}
        for t in activated_stracks:
            active_now[t.track_id] = t
        active_now = list(active_now.values())

        init_iou_thr = getattr(self, "init_iou_suppress", None)
        if init_iou_thr is None:
            print("Warn, init not found")
            init_iou_thr = 0.8

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue

            det_box = STrack.tlwh_to_tlbr(track.tlwh)

            if road_mask is not None and self.use_hard_road_filter:
                if not self._is_on_road(det_box, road_mask):
                    continue

            max_iou = 0.0
            for at in active_now:
                at_box = at.tlbr
                max_iou = max(max_iou, _iou(det_box, at_box))
                if max_iou >= init_iou_thr:
                    break

            if max_iou < init_iou_thr:
                track.activate(self.kalman_filter, self.frame_id)
                activated_stracks.append(track)

        # ------------------------------------------------------------
        # 10) 丢失轨迹管理：只在拓扑生命周期开关打开时动态调整
        # ------------------------------------------------------------
        for track in self.lost_stracks:
            recently_occluded = (
                track.was_recently_occluded and
                (self.frame_id - track.last_occluded_frame <= 40)
            )

            dynamic_max_time = self.max_time_lost

            if road_mask is not None and self.use_topology_lifecycle:
                last_box = track.tlbr
                cx = int((last_box[0] + last_box[2]) / 2)
                cy = int((last_box[1] + last_box[3]) / 2)

                cx = np.clip(cx, 0, img_w - 1)
                cy = np.clip(cy, 0, img_h - 1)

                is_near_image_edge = (
                    cx < self.edge_margin or cx > (img_w - self.edge_margin) or
                    cy < self.edge_margin or cy > (img_h - self.edge_margin)
                )

                dist_map = self._get_road_dist_map(road_mask)
                is_near_mask_edge = dist_map[cy, cx] > 10 if dist_map is not None else False

                if is_near_image_edge or is_near_mask_edge:
                    dynamic_max_time = 3
                else:
                    dynamic_max_time = int(self.max_time_lost * 1.5)

            if not recently_occluded and (self.frame_id - track.end_frame > dynamic_max_time):
                track.mark_removed()
                removed_stracks.append(track)

        # ------------------------------------------------------------
        # 11) 收尾
        # ------------------------------------------------------------
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks

    

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

