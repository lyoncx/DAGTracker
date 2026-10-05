from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from lib.loss.losses import FocalLoss
from lib.loss.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from lib.utils1.decode import ctdet_decode
from lib.utils1.utils import _sigmoid
from lib.utils1.debugger import Debugger
from lib.utils1.post_process import ctdet_post_process
from lib.Trainer.base_trainer_points import BaseTrainer
import cv2


class CtdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(CtdetLoss, self).__init__()
        # 检查是否使用 MMLoss
        self.use_mmloss = getattr(opt, 'use_mmloss', False)
        self.mmloss_scale = getattr(opt, 'mmloss_scale', 10)
        self.mmloss_version = getattr(opt, 'mmloss_version', 'v3')
        
        if self.use_mmloss:
            from lib.loss.losses import FocalLossWithMMLoss
            self.crit = FocalLossWithMMLoss(
                use_mmloss=True, 
                mmloss_scale=self.mmloss_scale,
                mmloss_version=self.mmloss_version
            )
            print(f"Using Motion Margin Loss (MMLoss) with scale={self.mmloss_scale}, version={self.mmloss_version}")
        else:
            self.crit = FocalLoss()  # torch.nn.MSELoss()

        # self.crit = FocalLoss()  # torch.nn.MSELoss()
        # self.crit = torch.nn.MSELoss()
        self.crit_reg = RegL1Loss()  # RegLoss()
        self.crit_wh = torch.nn.L1Loss(reduction='sum')  # NormRegL1Loss() # RegWeightedL1Loss()
        self.opt = opt
        self.wh_weight = opt.wh_weight
        self.hm_weight = opt.hm_weight
        self.off_weight = opt.off_weight
        self.num_stacks = 1

    def forward(self, outputs, batch):
        hm_loss, wh_loss, off_loss, lasso_loss = 0, 0, 0, 0

        output = outputs[0]

        b,c,t,h,w = output['hm'].shape

        for it in range(t):
            if self.opt.hm_flag:
                # 检查是否有光流信息
                flow_score = None
                if self.use_mmloss and 'flow_score' in batch:
                    # flow_score 的形状是 [B, T-1, H, W]，其中 T-1 = seqLen-1 = 19
                    # 输出热图的形状是 [B, C, T, H, W]，其中 T = seqLen = 20
                    # 光流对应相邻帧对：帧0->1, 1->2, ..., 18->19
                    flow_data = batch['flow_score']
                    
                    # 处理不同的形状
                    if len(flow_data.shape) == 4:
                        # [B, T-1, H, W] - 光流有 T-1 个时间步
                        # 对于时间步 it，使用对应的光流
                        # it=0: 使用 flow_data[:, 0] (帧0->1的光流)
                        # it=1: 使用 flow_data[:, 1] (帧1->2的光流)
                        # ...
                        # it=18: 使用 flow_data[:, 18] (帧18->19的光流)
                        # it=19: 使用 flow_data[:, 18] (最后一帧，使用最后一个光流)
                        if it < flow_data.shape[1]:
                            # 正常情况：it 在 [0, T-2] 范围内
                            flow_score = flow_data[:, it]  # [B, H, W]
                        elif it == flow_data.shape[1]:
                            # it = T-1，使用最后一个光流
                            flow_score = flow_data[:, -1]  # [B, H, W]
                        else:
                            # it > T-1，不应该发生，但使用最后一个光流
                            flow_score = flow_data[:, -1]  # [B, H, W]
                    elif len(flow_data.shape) == 3:
                        # [B, H, W] - 假设对所有时间步使用相同的光流
                        flow_score = flow_data
                    else:
                        # 形状不匹配，跳过 MMLoss（不打印警告，避免日志过多）
                        flow_score = None
                
                if flow_score is not None:
                    # 使用 MMLoss
                    hm_loss += self.crit(
                        output['hm'][:,:,it].contiguous(), 
                        batch['hm'][:,:,it],
                        flow_score=flow_score
                    ) / self.num_stacks
                else:
                    # 使用标准 Focal Loss
                    hm_loss += self.crit(
                        output['hm'][:,:,it].contiguous(), 
                        batch['hm'][:,:,it]
                    ) / self.num_stacks
                    
            # if self.opt.hm_flag:
            #     hm_loss += self.crit(output['hm'][:,:,it].contiguous(), batch['hm'][:,:,it]) / self.num_stacks

            if self.opt.wh_flag:
                wh_loss += self.crit_reg(
                    output['wh'][:,:,it].contiguous(), batch['reg_mask'][:, it],
                    batch['ind'][:, it], batch['wh'][:,it]) / self.num_stacks

            if self.opt.off_flag:
                off_loss += self.crit_reg(output['reg'][:,:, it].contiguous(), batch['reg_mask'][:,it],
                                          batch['ind'][:,it], batch['reg'][:,it])

        hm_loss = hm_loss/t
        wh_loss = wh_loss/t
        off_loss = off_loss/t

        loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + \
               self.off_weight * off_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}

        return loss, loss_stats


class CtdetTrainer_points(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(CtdetTrainer_points, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        if opt.off_flag:
            loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        else:
            loss_states = ['loss', 'hm_loss', 'wh_loss']
        loss = CtdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=opt.cat_spec_wh, K=opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((
                                   img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
                                           dets[i, k, 4], img_id='out_pred')

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 4] > opt.center_thresh:
                    debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
                                           dets_gt[i, k, 4], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = ctdet_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]