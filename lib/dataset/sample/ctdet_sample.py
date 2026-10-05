from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import json
import cv2
import os
from lib.utils1.image import flip, color_aug
from lib.utils1.image import get_affine_transform, affine_transform
from lib.utils1.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.utils1.image import draw_dense_reg
import math
from lib.utils1.augmentations import Augmentation_st
from lib.dataset.data_aug.data_aug import RandomSampleCrop
import torch.utils.data as data

FLOW_MAGIC = b'PIEH'


def _safe_load_coords(path):
    """从路径加载坐标文件，若不存在/为空则返回 None。"""
    if not path or not os.path.exists(path):
        return None
    if os.path.getsize(path) == 0:
        return None
    try:
        coords = np.loadtxt(path)
        if coords.size == 0:
            return None
        return coords.reshape(-1, 6)
    except Exception as e:
        warnings.warn(f"加载坐标失败 {path}: {e}")
        return None


def _coords_to_anns(coords):
    """将坐标数组转换为 ctdet 训练所需的 anns 结构。"""
    if coords is None:
        return []
    return [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
             coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]


def _read_flo_file(flow_path):
    """读取 .flo 文件并返回 [H, W, 2] 光流."""
    with open(flow_path, 'rb') as f:
        header = f.read(4)
        if header != FLOW_MAGIC:
            raise ValueError(f"{flow_path} 不是有效的 .flo 文件（缺少 PIEH 头）")
        width = np.fromfile(f, np.int32, 1).squeeze()
        height = np.fromfile(f, np.int32, 1).squeeze()
        data = np.fromfile(f, np.float32, width * height * 2)
    return data.reshape((height, width, 2))


def _load_flow_magnitude(flow_path):
    """加载光流强度，兼容 .npy/.flo."""
    ext = os.path.splitext(flow_path)[1].lower()
    if ext == '.npy':
        return np.load(flow_path)
    if ext == '.flo':
        flow = _read_flo_file(flow_path)
        return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    raise ValueError(f"不支持的光流格式: {flow_path}")


# 缓存文件查找结果，避免重复的文件系统访问
_flow_file_cache = {}

def _resolve_flow_file(flow_dir, split, video_id, frame_id):
    """在多种目录/后缀组合中查找光流文件（带缓存）."""
    frame_name = f"{frame_id:06d}"
    cache_key = (flow_dir, split, video_id, frame_name)
    
    # 检查缓存
    if cache_key in _flow_file_cache:
        return _flow_file_cache[cache_key]
    
    # 查找文件
    base_candidates = []
    if split:
        base_candidates.append(os.path.join(flow_dir, split, video_id, frame_name))
    base_candidates.append(os.path.join(flow_dir, video_id, frame_name))
    for base in base_candidates:
        for ext in ('.npy', '.flo'):
            candidate = base + ext
            if os.path.exists(candidate):
                _flow_file_cache[cache_key] = candidate
                return candidate
    
    # 文件不存在，缓存 None
    _flow_file_cache[cache_key] = None
    return None

class CTDetDataset(data.Dataset):

    def get_im_ids(self, img_id):
        img_info = self.coco.loadImgs(ids=[img_id])[0]
        # video_id = img_info['video_id']
        video_id = img_info['video_id']
        video_frame_id = img_info['video_frame_id']
        video_len = self.video_lens[video_id]
        video_info = self.video_to_images[video_id]

        # 处理边界情况
        if video_frame_id + self.seqLen - 1 > video_len:
            start_frame_id = video_len - self.seqLen + 1
        else:
            start_frame_id = video_frame_id

        im_ids = []
        for i in range(self.seqLen):
            curr_frame_id = start_frame_id + i
            if curr_frame_id in video_info:
                img_id_i = video_info[curr_frame_id]['id']
                im_ids.append(img_id_i)
            else:
                # 处理缺失的帧，例如跳过或填充
                print(f"Frame {curr_frame_id} not found in video {video_id}")
                continue

        return im_ids

    def get_heatmap(self, num_classes, output_h, output_w, c, s, bbox_tol, cls_id_tol):
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        hm_mask = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
        draw_gaussian = draw_umich_gaussian
        num_objs = min(len(bbox_tol), self.max_objs)
        gt_det = []
        for k in range(num_objs):
            bbox = bbox_tol[k]
            cls_id = int(cls_id_tol[k])
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:4] = affine_transform(bbox[2:], trans_output)

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            h = np.clip(h, 0, output_h - 1)
            w = np.clip(w, 0, output_w - 1)
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct[0] = np.clip(ct[0], 0, output_w - 1)
                ct[1] = np.clip(ct[1], 0, output_h - 1)
                ct_int = ct.astype(np.int32)

                draw_gaussian(hm[cls_id], ct_int, radius)

                hm_mask[cls_id, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 1

                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                # if self.dense_wh:
                #     draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret_hm = {}
        ret_hm['hm'] = hm
        ret_hm['hm_mask'] = hm_mask
        ret_hm['reg_mask'] = reg_mask
        ret_hm['ind'] = ind
        ret_hm['wh'] = wh
        ret_hm['reg'] = reg
        return ret_hm

    def get_single(self, img_id):
        #get info
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        #read the images
        im = cv2.imread(self.img_dir + file_name)
        ###
        if self.opt.sup_mode == 0:  #directly load the annotated labels
            anns0 = self.coco.loadAnns(ids=ann_ids)
            anns1 = [[i['bbox'][0], i['bbox'][1], i['bbox'][0]+i['bbox'][2], i['bbox'][1]+i['bbox'][3],
                     self.cat_ids[i['category_id']], i['track_id']
                     ] for i in anns0]
        elif self.opt.sup_mode == 1:  #load the generated unfilt labels
            coords = np.loadtxt(self.img_dir+file_name.replace('images', 'lrsd').replace('img1', 'coords_unfilt').replace('.jpg','.txt')).reshape(-1,6)
            anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                     coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
        elif self.opt.sup_mode == 2: #load the generated filt labels
            coords = np.loadtxt(self.img_dir+file_name.replace('images', 'lrsd').replace('img1', 'coords_filt').replace('.jpg','.txt')).reshape(-1,6)
            anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                     coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
        elif self.opt.sup_mode == 3: #load the generated updated labels
            coords_path = os.path.join(
                    self.img_dir,
                    'lrsd',
                    file_name.replace('img1', 'coords_update').replace('.jpg', '.txt')
                )
            # print("coords_path:", coords_path)
            coords = np.loadtxt(coords_path).reshape(-1, 6)
            anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                      coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
        else:
            raise Exception('Not a valid sup_mode!!!!')
        #augmentation
        if self.aug is not None:
            im, anns = self.apply_aug(im ,anns1)
        #get gray image
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = np.expand_dims(im_gray, 0)
        ##normalization
        im = (im - self.mean) / self.std
        im = im.transpose(2, 0, 1)
        ##get hm
        _, input_h, input_w = im.shape
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        c = np.array([input_w / 2., input_h / 2.], dtype=np.float32)
        s = max(input_h, input_w) * 1.0
        #
        bbox_temp = anns[:,:4].tolist()
        cls_id_temp = anns[:,4].tolist()
        ret_single = self.get_heatmap(num_classes, output_h, output_w, c, s,bbox_temp, cls_id_temp)
        #####
        ret_single.update({'input': im, 'input_gray': im_gray*255})
        return ret_single

    def get_multi(self, img_id):
        multi_ids = self.get_im_ids(img_id)
        ret_multi = {'input': [], 'input_gray': [], 'hm': [], 'hm_mask': [],'reg_mask': [], 'ind': [], 'wh': []}
        ret_multi.update({'reg': []})
        #choose which frame as input
        img_chosen_id = self.seqLen // 2
        for c_id  in multi_ids:
            ret_tmp = self.get_single(c_id)
            for k,v in ret_tmp.items():
                if k == 'im_ids' or k == 'file_name' or k=='meta':
                    continue
                else:
                    if  k=='reg_mask' or k=='ind' or k=='wh' or k=='reg':
                        ret_multi[k].append(np.expand_dims(v, 0))
                    elif k=='hm' or k=='hm_mask' :
                        ret_multi[k].append(np.expand_dims(v, 1))
                    else:
                        ret_multi[k].append(np.expand_dims(v,1))
        for k,v in ret_multi.items():
            if k == 'reg_mask' or k == 'ind' or k == 'wh' or k == 'reg':
                v1 = np.concatenate(v, axis=0)
                # v1 = v1[img_chosen_id]
            elif k == 'hm' or k == 'hm_mask':
                v1 = np.concatenate(v, axis=1)
                # v1 = v1[:,img_chosen_id]
            else:
                v1 = np.concatenate(v, axis=1)
            ret_multi[k] = v1
        
        # 加载光流数据（如果启用 MMLoss 或将光流作为输入，或使用光流分支）
        if ((hasattr(self.opt, 'use_mmloss') and self.opt.use_mmloss) or
            (hasattr(self.opt, 'use_flow_input') and self.opt.use_flow_input) or
            (hasattr(self.opt, 'use_flow_branch') and self.opt.use_flow_branch)):
            flow_scores = []
            flow_dir = getattr(self.opt, 'flow_dir', '')
            
            if flow_dir:
                # 获取视频ID和帧ID信息
                img_info = self.coco.loadImgs(ids=[img_id])[0]
                video_id = img_info['video_id']
                video_frame_id = img_info['video_frame_id']
                video_len = self.video_lens[video_id]
                
                # 计算起始帧ID
                if video_frame_id + self.seqLen - 1 > video_len:
                    start_frame_id = video_len - self.seqLen + 1
                else:
                    start_frame_id = video_frame_id
                
                # 加载每一对相邻帧的光流
                for t in range(self.seqLen - 1):
                    curr_frame_id = start_frame_id + t
                    next_frame_id = start_frame_id + t + 1
                    
                    flow_file = _resolve_flow_file(flow_dir, self.split, video_id, curr_frame_id)
                    if flow_file:
                        try:
                            flow_magnitude = _load_flow_magnitude(flow_file)
                            flow_scores.append(flow_magnitude)
                        except Exception as e:
                            print(f"Warning: Failed to load flow file {flow_file}: {e}")
                            # 如果加载失败，使用零填充
                            if len(flow_scores) > 0:
                                flow_scores.append(np.zeros_like(flow_scores[0]))
                            else:
                                # 如果这是第一个，需要知道图像尺寸
                                # 从输入图像获取尺寸
                                h, w = ret_multi['input'].shape[2], ret_multi['input'].shape[3]
                                flow_scores.append(np.zeros((h, w), dtype=np.float32))
                    else:
                        # 如果文件不存在，使用零填充
                        if len(flow_scores) > 0:
                            flow_scores.append(np.zeros_like(flow_scores[0]))
                        else:
                            h, w = ret_multi['input'].shape[2], ret_multi['input'].shape[3]
                            flow_scores.append(np.zeros((h, w), dtype=np.float32))
                
                if len(flow_scores) > 0:
                    # 堆叠为 [T-1, H, W]
                    flow_stack = np.stack(flow_scores, axis=0).astype(np.float32)
                    ret_multi['flow_score'] = flow_stack

                    # 为光流分支提供完整的时间序列光流数据，保留时间动态信息
                    # 光流有 T-1 帧（相邻帧之间的光流），图像序列有 T 帧
                    # 需要将 [T-1, H, W] 扩展为 [T, H, W]，通过复制首尾帧对齐
                    # 方法：第一帧复制一次，最后一帧也复制一次
                    T_seq = self.seqLen  # 图像序列长度
                    T_flow = flow_stack.shape[0]  # 光流帧数 (T-1)
                    if T_flow == T_seq - 1:
                        # 在开头和结尾各复制一帧，得到 [T, H, W]
                        flow_expanded = np.zeros((T_seq, flow_stack.shape[1], flow_stack.shape[2]), 
                                                dtype=np.float32)
                        flow_expanded[0] = flow_stack[0]  # 第一帧：复制第一个光流
                        flow_expanded[1:-1] = flow_stack[1:]  # 中间帧：使用原始光流
                        flow_expanded[-1] = flow_stack[-1]  # 最后一帧：复制最后一个光流
                    else:
                        # 如果数量不匹配，使用插值或复制（fallback）
                        flow_expanded = np.zeros((T_seq, flow_stack.shape[1], flow_stack.shape[2]), 
                                                dtype=np.float32)
                        # 简单复制策略：将 T-1 帧光流平均分布到 T 帧
                        indices = np.linspace(0, T_flow - 1, T_seq, dtype=np.int32)
                        flow_expanded = flow_stack[indices]
                    
                    # flow_branch 期望 batch['flow'] 形状为 [B, 1, H, W, T] 或 [B, 1, H, W]
                    # 这里提供完整时间序列 [T, H, W]，在模型中处理
                    ret_multi['flow'] = flow_expanded.astype(np.float32)
                    
        return ret_multi

    def get_aug(self, annos=None):
        s0 = np.random.choice(np.arange(0.9, 1.2, 0.1))
        c = np.array([self.resolution_ori[1] / 2., self.resolution_ori[0] / 2.], dtype=np.float32)
        s = max(self.resolution_ori[0], self.resolution_ori[1]) * s0
        self.trans_ori = get_affine_transform(c, s, 0, [self.resolution_ori[1], self.resolution_ori[0]])  # random scale
        #######crop target region
        annos_t = []
        if annos is not None:
            for anno in annos:
                anno = affine_transform(anno, self.trans_ori)
                if anno[0]>0 and anno[1]>0 and anno[0]<self.resolution_ori[0] and anno[1]<self.resolution_ori[1]:
                    annos_t.append(anno)
        if len(annos_t)>0:
            id = np.random.randint(0, len(annos_t))
            anno_crop = annos_t[id][0:2]
            anno_crop = anno_crop[::-1]
        else:
            anno_crop= None
        #######
        self.crop = RandomSampleCrop(self.resolution_ori, self.resolution, anno_coord=anno_crop)  # random crop
        self.aug = Augmentation_st()  # random mirror
        self.color_aug = color_aug(self._eig_val, self._eig_vec)  # random color

    def apply_aug(self, im, anns):
        num_objs = len(anns)
        # random scale
        try:
            im = cv2.warpAffine(im, self.trans_ori,
                                (self.resolution_ori[1], self.resolution_ori[0]),
                                flags=cv2.INTER_LINEAR)
        except:
            a=1
        bboxes = []
        for k in range(num_objs):
            bbox = anns[k]
            ##random scale
            bbox[:2] = affine_transform(bbox[:2], self.trans_ori)
            bbox[2:4] = affine_transform(bbox[2:4], self.trans_ori)
            bboxes.append(bbox)
        # random crop
        im, anns_new = self.crop(im, np.array(bboxes).reshape(-1, 6))
        #random mirror
        im, bbox_tol, cls_id_tol = self.aug(im, anns_new[:,:4], anns_new[:,4:])
        anns_new[:, :4] = bbox_tol
        #random color
        im = (im.astype(np.float32) / 255.)
        # im = self.color_aug(im)
        return im, anns_new

    def __getitem__(self, index):
        ######get image id
        img_id = self.images[index]
        ######get params
        self.down_ratio = self.opt.down_ratio
        self.max_objs = self.opt.K
        ######aug or not
        if self.split == 'train':
            file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            # read anns
            if self.opt.sup_mode == 0:  # directly load the annotated labels
                anns0 = self.coco.loadAnns(ids=ann_ids)
                anns1 = [[i['bbox'][0], i['bbox'][1], i['bbox'][0] + i['bbox'][2], i['bbox'][1] + i['bbox'][3],
                          self.cat_ids[i['category_id']], i['track_id']
                          ] for i in anns0]
            elif self.opt.sup_mode == 1:  # load the generated unfilt labels
                coords = np.loadtxt(
                    self.img_dir + file_name.replace('images', 'lrsd').replace('img1', 'coords_unfilt').replace('.jpg',
                                                                                                                '.txt')).reshape(
                    -1, 6)
                anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                          coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
            elif self.opt.sup_mode == 2:  # load the generated filt labels
                coords = np.loadtxt(
                    self.img_dir + file_name.replace('images', 'lrsd').replace('img1', 'coords_filt').replace('.jpg',
                                                                                                              '.txt')).reshape(
                    -1, 6)
                anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                          coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
            elif self.opt.sup_mode == 3:  # load the generated updated labels
                coords_path = os.path.join(
                    self.img_dir,
                    'lrsd',
                    file_name.replace('img1', 'coords_update').replace('.jpg', '.txt')
                )
                # print("coords_path:", coords_path)
                coords = np.loadtxt(coords_path).reshape(-1, 6)
                anns1 = [[coords[i, 0], coords[i, 1], coords[i, 2], coords[i, 3],
                          coords[i, 4], coords[i, 5]] for i in range(coords.shape[0])]
            else:
                raise Exception('Not a valid sup_mode!!!!')
            self.get_aug(anns1)
            # self.get_aug()
        else:
            self.aug = None
        #####switch mode
        if self.opt.data_mode == 'single':
            ret = self.get_single(img_id)
        elif self.opt.data_mode == 'multi':
            self.seqLen = self.opt.seqLen
            ret = self.get_multi(img_id)
        else:
            raise Exception('Not a valid data mode!!!')
        return img_id, ret