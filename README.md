<div align="center">
    <h2>SpaRTAN: Spatial Reinforcement Token-based Aggregation Network for Visual Recognition (IJCNN 2025)</h2>
</div>

<p align="center">
    <a href="https://github.com/henry-pay/SpaRTAN/blob/main/LICENSE" alt="license">
        <img src="https://img.shields.io/badge/license-Apache--2.0-%23B7A800" />
    </a>
</p>

We propose **SpaRTAN**, a lightweight architectural design that improves spatial- and channel-wise information processing. It shows consistent efficiency and competitive performance when benchmarked against ImageNet and COCO datasets. 

This repository contains the PyTorch implementation for SpaRTAN (IJCNN 2025).

## Image Classification

### 1. Installation

In this section, we provide instructions for ImageNet classification experiments.

#### 1.1 Dependency Setup

Create a new conda environment
```
conda create -y -n spartan python=3.12
conda activate spartan
```

Install [Pytorch](https://pytorch.org/)>=2.4.0, [torchvision](https://pytorch.org/vision/stable/index.html)>= 0.19.0 following official instructions. For example:
```
conda install -y pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Clone this repo and install required packages.
```
git clone https://github.com/henry-pay/SpaRTAN.git
conda install -y timm 
conda install -y fvcore iopath -c fvcore -c iopath 
conda install -y hydra-core 
conda install -y matplotlib tensorboard seaborn
pip install grad-cam
```

#### 1.2 Dataset Preparation

Download the [ImageNet-1k](http://image-net.org/) classification dataset and structure the data as follows. You can extract ImageNet with this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

Place the imagenet dataset under data directory within the repository.
```
│SpaRTAN/
├──data/
│  ├── imagenet/
│  │   ├──train/
│  │   ├──val/
│  │   ├──images/
├──src/
```
Note that the `images` directory contains selected images for visualization.

### 2. Training

We provide ImageNet-1k training commands here.

Taking SpaRTAN-T as an example, you can use the following command to run the experiment on a single machine (4 GPUs)
```
OMP_NUM_THREADS=8 torchrun --nproc-per-node=4 src/main.py
```

- Batch size scaling. The effective batch size is equal to ``--nproc-per-node`` * ``batch_size`` (which is specified in the [dataset config](src/config/data/imagenet.yaml)). In the provided config file, the effective batch size is ``4*512=2048``. Running on machine, we can reduce ``batch_size`` and set ``use_amp`` flag in the [config](src/config/config.yaml) to avoid OOM issues while keeping the total batch size unchanged.
- OMP_NUM_THREADS is the easiest switch that can be used to accelerate computations. It determines number of threads used for OpenMP computations. Details can be found in [documentation](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

To train other SpaRTAN variants, parameters within the [config](src/config/) need to be changed.

## Object Detection

The experiment is carried out using [RT-DETRv1](https://github.com/lyuwenyu/RT-DETR/tree/main). Please refer to the corresponding repository for installation and dataset preparation instructions. By replacing the backbone model in RT-DETR with SpaRTAN, the training experiments can be run based on the given instructions in RT-DETR.

## License

This project is licensed under the [Apache 2.0 License](LICENSE)

## Citation

If you find this repository helpful, please consider citing:
```

```

<p align="right">(<a href="#top">back to top</a>)</p>