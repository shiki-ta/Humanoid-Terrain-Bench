# [Multi-Terrain Humanoid Locomotion Challenge in Human-Robot-Scene Interaction and Collaboration (ICCV 2025 Workshop)](https://human-robot-scene.github.io/Terrain-Challenge/)

![Workshop](images/workshop.png)

## Introduction

This package(challenging_terrain) contains 9 types of terrains (which will be continuously expanded in the future) and a large number of terrains arranged in various combinations. The basic module configuration code based on Legged Gym is provided, allowing users to achieve plug and play of terrain modules within their existing Legged Gym framework. 

This code is compatible with various robots, including but not limited to humanoid robots such as Unitree G1, Unitree H1-2, Fourier GR1-T2, Fourier GRX-N1,which will be continuously added in the future. 

Provided an online data collection module that can store trained policies in dataset format. The evaluation module is embedded in the code and only requires one parameter to quantitatively evaluate the trained policy indicators.

### Installation ###
```bash
conda create -n terrain python=3.8
conda activate terrain
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   #or cu113,cu115,cu121, based on your cuda version

git clone https://github.com/shiki-ta/Humanoid-Terrain-Bench.git
cd Humanoid-Terrain-Bench
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
cd isaacgym/python && pip install -e .
cd rsl_rl && pip install -e .
cd legged_gym && pip install -e .
cd challenging_terrain && pip install -e .
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
```

### Usage ###
`cd legged_gym/scripts`
1. Train base policy:  
```
python train.py --exptid=test
```

2. Training Recovery:
```
python train.py --exptid=test-1  --resume --resumeid=test --checkpoint=50000
```

3. Play base policy:
```
python play.py --exptid=test
```

### Arguments ###
- --exptid: string,  to describe the run. 
- --device: can be `cuda:0`, `cpu`, etc.
- --checkpoint: the specific checkpoint you want to load. If not specified load the latest one.
- --resume: resume from another checkpoint, used together with `--resumeid`.
- --seed: random seed.
- --no_wandb: no wandb logging.
- --save: make dataset

### Acknowledgement ###

[legged_gym](https://github.com/leggedrobotics/legged_gym)

[Isaac Gym](https://developer.nvidia.com/isaac-gym)

[extreme parkour](https://github.com/chengxuxin/extreme-parkour)

### Citation
If you found any part of this code useful, please consider citing:
```
@article{fan2025one,
  title={One Policy but Many Worlds: A Scalable Unified Policy for Versatile Humanoid Locomotion},
  author={Fan, Yahao and Gui, Tianxiang and Ji, Kaiyang and Ding, Shutong and Zhang, Chixuan and Gu, Jiayuan and Yu, Jingyi and Wang, Jingya and Shi, Ye},
  journal={arXiv preprint arXiv:2505.18780},
  year={2025}
}
```