# DAGTracker: Satellite Video Multi-Object Tracking
![DAGTracker framework](readme/framework.png)
This repository contains the implementation and supplementary resources for **DAGTracker**, a three-stage framework for multi-object tracking of small and densely distributed vehicles in satellite videos.

The accompanying paper presents a detection-association-global optimization pipeline designed for dim targets, motion blur, dense traffic, occlusion, and fragmented trajectories.

## Highlights

- **VDD-VEH dataset:** a satellite-video vehicle MOT benchmark with motion annotations.
- **M-3DSD detector:** Motion-guided 3D Sparse Detection. Frame differencing constructs sparse voxels, while optical-flow-derived motion information provides motion-margin supervision through MMLoss.
- **Road-prior constrained association:** soft and hard road constraints suppress background detections that are inconsistent with road topology.
- **Global trajectory optimization:** graph-based optimization combines spatio-temporal geometry, motion-direction consistency, and temporally decayed appearance features to reconnect fragmented tracklets.


## Installation

The code was developed and tested with the following configuration:

- Ubuntu 20.04
- Python 3.10
- PyTorch 2.0.1
- Torchvision 0.15.2
- CUDA 11.7
- 2x NVIDIA GeForce RTX 4090

You can follow [DSFNet](https://github.com/ChaoXiao12/Moving-object-detection-DSFNet) to build the environment.

The supplied scripts may require minor adjustments for newer PyTorch, CUDA, or GPU driver versions.

## Data Preparation

You can now download the complete datasets directly from [BaiduYun](https://pan.baidu.com/s/1h2xqaZ2V1D1M_65ZGcWHoA?pwd=aqx2)(Sharing code: aqx2).

Download the VDD-VEH dataset and arrange it under the project data directory. A typical layout is:

```text
data/
└── VDD-VEH/
    ├── train/
    ├── val/
    ├── test/
    └── annotations/
```
The trained weight is available at [BaiduYun](https://pan.baidu.com/s/1DoDF5xDwysiigsWPLa3w3g?pwd=w7cm)(Sharing code: w7cm). You can download the model and put it to the weights folder.
## Training and Evaluation

DAGTracker uses `train_sp_update.py` as the training entry point and `sp_centerDet_minus` as the detection model. Before running the commands, update the paths below for your machine:

FLOW_DIR must contain the precomputed optical-flow results required by MMLoss. You can generate these files by following the instructions in [EMD-Flow](https://github.com/gddcx/EMD-Flow).

### Training with MMLoss

The following command reproduces the reported unsupervised iterative MMLoss training setup:

```bash
python train_sp_update.py --task ctdet_points --model_name sp_centerDet_minus --layers 3 --gpus 0,1 --datasetname rs_car --data_mode multi --data_dir "${DATA_DIR}" --exp_name DAGTracker --sup_mode 0 --unsup_iter 10 --off_flag True --down_ratio 1 --seqLen 20 --conf_filtered 0.2 --val_intervals 2 --lr 1.25e-4 --lr_step 30,45 --num_epochs 55 --batch_size 8 --data_sampling 5 --save_dir "${OUTPUT_DIR}" --use_mmloss --mmloss_scale 3.5 --mmloss_version v3 --flow_dir "${FLOW_DIR}"
```

To resume training from a checkpoint, append the following options and replace the checkpoint path:

```bash
  --load_model /path/to/checkpoint/model_last.pth --resume True
```

The training configuration uses two GPUs (`0,1`). Change `--gpus` and `--batch_size` according to the available hardware.

### Testing a Trained Model

Set `MODEL_PATH` to the trained checkpoint and run:

```bash
python test.py --task ctdet_points --model_name sp_centerDet_minus --layers 3 --gpus 0 --datasetname rs_car --data_mode multi --data_dir "${DATA_DIR}" --sup_mode 0 --off_flag True --down_ratio 1 --seqLen 20 --load_model "${MODEL_PATH}" --use_mmloss --mmloss_scale 3.5 --mmloss_version v3 --flow_dir "${FLOW_DIR}"
```

The test command uses one GPU and loads the MMLoss-trained checkpoint. Make sure there is a space before each continued option; in particular, `--load_model` and `--use_mmloss` must remain separate arguments.

During training, frame differencing constructs sparse voxel locations and optical flow provides motion-margin supervision. During inference, sparse voxel construction remains active, while optical-flow-based MMLoss supervision is not required unless the released test implementation explicitly uses flow features.

## Quantitative Results

Overall results on four satellite-video MOT benchmarks are listed below. Higher is better for MOTA, IDF1, and MT; lower is better for ML, FP, FN, and IDs.

| Dataset | MOTA ↑ | IDF1 ↑ | MT ↑ | ML ↓ | FP ↓ | FN ↓ | IDs ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VDD-VEH | 57.5% | 70.2% | 32 | 16 | 462 | 2,836 | 59 |
| VISO | 67.8% | 76.4% | 64 | 29 | 864 | 12,966 | 139 |
| SAT-MTB | 60.0% | 78.5% | 40 | 10 | 599 | 1,808 | 23 |
| SDM-Car | 55.2% | 64.8% | 58 | 18 | 1,249 | 5,107 | 247 |

On VDD-VEH, DAGTracker reaches 57.5% MOTA and 70.2% IDF1, while maintaining 25.7 FPS under the reported evaluation setting.

## Citation

If you find this project useful, please consider citing the following works.

### DAGTracker

```bibtex
% Replace this entry with the final bibliographic information of your paper.
@article{dagtracker,
  title   = {DAGTracker: A Three-Stage Framework for Satellite-Video Multi-Object Tracking},
  author  = {Chengxin Liang, Huyi Song, Jia Shao, Du Bo},
  journal = {TBD},
  year    = {2026}
}
```
## Acknowledgements

This project was developed with reference to the implementations of
[MM-Tracker](https://github.com/YaoMufeng/MMTracker) and
[HiEUM](https://github.com/ChaoXiao12/Moving-object-detection-in-satellite-videos-HiEUM)and
[FastTracker](https://github.com/Hamidreza-Hashempoor/FastTracker).
Parts of the training pipeline, motion-margin loss, and detection
implementation were adapted from these projects. We sincerely thank
the authors for releasing their code and for their valuable contributions.
