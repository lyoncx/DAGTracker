from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
from datetime import datetime

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('--task', default='ctdet_points',
                                 help='task name.  ctdet_points |  ctdet ')
        self.parser.add_argument('--exp_name', default='unsupervised_iterative_layers_3_',
                                 help='name of the experiments.')
        self.parser.add_argument('--layers', type=int, default=3,
                                 help='use decomp model or not.')
        self.parser.add_argument('--model_name', default='sp_centerDet_minus',
                                 help='name of the model.')
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', type=bool, default=False,
                                 help='resume an experiment.')
        self.parser.add_argument('--down_ratio', type=int, default=1,
                                 help='output stride. Currently only supports for 1.')
        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 4.')
        self.parser.add_argument('--lr_step', type=str, default='30,45', #30,45
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=55,  #55
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=6,
                                 help='batch size')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--seqLen', type=int, default=20,
                                 help='number of images for per sample. Currently supports 5.')

        self.parser.add_argument('--snapshot_file', type=int, default=1000,
                                 help='1000 iterations per epoch.')
        self.parser.add_argument('--mode', type=str, default='ms',
                                 help='lstm mode. m for move | s for space | ms for both.')


        # test
        self.parser.add_argument('--nms', action='store_true',
                                 help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=128,
                                 help='max number of output objects. top_k')
        self.parser.add_argument('--test_large_size', type=bool, default=False,
                                 help='whether or not to test image size of 1024. Only for test.')
        self.parser.add_argument('--show_results', type=bool, default=True,
                                 help='whether or not to show the detection results. Only for test.')
        self.parser.add_argument('--save_track_results', type=bool, default=False,
                                 help='whether or not to save the tracking results of sort. Only for testTrackingSort.')

        # save
        self.parser.add_argument('--save_dir', type=str, default='./weights',
                                 help='savepath of model.')
        self.parser.add_argument('--track_results_dir', type=str, default='./results/',
                                 help='savepath of results.')
        self.parser.add_argument('--auto_test', type=bool, default=False,
                                 help='whether or not this is an auto test for optuna.')

        # dataset
        self.parser.add_argument('--data_mode', type=str, default='multi',
                                 help='dataset name.')
        self.parser.add_argument('--datasetname', type=str, default='viso_rs_car',
                                 help='dataset name.')
        self.parser.add_argument('--data_dir', type=str, default= '/media/wellwork/L/xc/datasets/RsCarData/',
                                 help='path of dataset.')
        self.parser.add_argument('--data_sampling', type=int, default=1,
                                 help='data_sampling.')

        #update_label
        self.parser.add_argument('--sup_mode', type=int, default=0,
                                 help='supervion mode.0 for annotated labels.|  1 for unfilt generated labels. | 2 for filt generated labels. | 3 for updated generated labels.')
        self.parser.add_argument('--unsup_iter', type=int, default=10,
                                 help='unsup iteration interval.')
        self.parser.add_argument('--conf_filtered', type=float, default=0.2,
                                 help='conf_filtered.')

        #loss
        self.parser.add_argument('--hm_flag', type=bool, default=True,
                                 help='offset brantch.')
        self.parser.add_argument('--hm_weight', type=float, default=1.0,
                                 help='wh weight in loss.')
        self.parser.add_argument('--wh_flag', type=bool, default=True,
                                 help='offset brantch.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1,
                                 help='wh weight in loss.')
        self.parser.add_argument('--off_flag', type=bool, default=True,
                                 help='offset brantch.')
        self.parser.add_argument('--off_weight', type=float, default=1.0,
                                 help='offset weight in loss.')

        # tracking
        self.parser.add_argument('--track', type=bool, default=True,
                                 help='whether or not to do tracking.')
        self.parser.add_argument('--conf_thres', type=float, default=0.3,
                                 help='confidence threshold to filter out boxes.')
        self.parser.add_argument('--track_buffer', type=int, default=30,
                                 help='tracking buffer')
        self.parser.add_argument('--track_thresh', type=float, default=0.6,
                                 help='track_thresh')    
        self.parser.add_argument('--match_thresh', type=float, default=0.8,
                                 help='match_thresh')
        self.parser.add_argument('--min_box_area', type=int, default=100,
                                 help='min_box_area')
        self.parser.add_argument('--reset_velocity_offset_occ', type=int, default=5,
                                 help='reset_velocity_offset_occ')
        self.parser.add_argument('--reset_pos_offset_occ', type=int, default=3,
                                 help='reset_pos_offset_occ')
        self.parser.add_argument('--enlarge_bbox_occ', type=float, default=1.2,
                                 help='enlarge_bbox_occ')
        self.parser.add_argument('--dampen_motion_occ', type=float, default=0.85,
                                 help='dampen_motion_occ')
        self.parser.add_argument('--active_occ_to_lost_thresh', type=int, default=15,
                                 help='active_occ_to_lost_thresh')    
        self.parser.add_argument('--init_iou_suppress', type=float, default=0.5,
                                 help='init_iou_suppress')  
        self.parser.add_argument('--use_global_optim', type=bool, default=False,
                                 help='use_global_optim')   
        
        # Motion Margin Loss (MMLoss) from MMTracker
        self.parser.add_argument('--use_mmloss', action='store_true',
                                 help='use motion margin loss from MMTracker')
        self.parser.add_argument('--mmloss_scale', type=float, default=10,
                                 help='scale factor for motion margin loss (default: 10)')
        self.parser.add_argument('--mmloss_version', type=str, default='v3',
                                 choices=['v1', 'v2', 'v3'],
                                 help='version of motion margin loss: v1 (flow_ldam), v2 (flow_ldam_v2), v3 (flow_ldam_v3, recommended)')
        self.parser.add_argument('--flow_dir', type=str, default='/home/liangcx/datasets/crop_datasets/flow',
                                 help='directory containing pre-computed optical flow files (optional)')
        
    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.dataName = opt.data_dir.split('/')[-2]

        return opt
