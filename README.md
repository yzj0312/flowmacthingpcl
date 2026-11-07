# EGIInet: Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion

This repository contains the official implementation for "Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion" (ECCV 2024) 

**[Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion](https://arxiv.org/abs/2407.02887), ECCV 2024**

## Introduction

This work is accepted by ECCV 2024. In comparison with previous methods that relied on the global semantics of input images, EGIInet efficiently combines the information from two modalities by leveraging the geometric nature of the completion task. Specifically, we propose an explicitly guided information interaction strategy supported by modal alignment for point cloud completion. First, we unified the encoding process to promote modal alignment. Second, we propose a novel explicitly guided information interaction strategy that could help the network identify critical information within images, thus achieving better guidance for completion. Extensive experiments demonstrate the effectiveness of our framework, and we achieved a new state-of-the-art (+16% CD over XMFnet) in benchmark datasets despite using fewer parameters than the previous methods.

## Get Started

### Requirement
- python >= 3.6
- PyTorch >= 1.8.0
- CUDA == 11.8
- easydict
- opencv-python
- transform3d
- h5py
- timm
- open3d
- tensorboardX
- ninja == 1.11.1
- torch-scatter
- einops

Install PointNet++ utils, Density-aware Chamfer Distance and Furthest Point Sampling.
```
cd /models/pointnet2_batch
python setup.py install

cd ../../metrics/CD/chamfer3D/
python setup.py install

cd ../../EMD/
python setup.py install

cd ../../utils/furthestPointSampling/
python setup.py install
```

The code has been trained and tested with Python 3.9, PyTorch 2.1.2 and CUDA 11.8 on Ubuntu 20.04 and RTX 3080Ti.

### Dataset
Download the ShapeNet-ViPC dataset
First, please download the [ShapeNetViPC-Dataset](https://pan.baidu.com/s/1NJKPiOsfRsDfYDU_5MH28A) (143GB, code: **ar8l**). Then run ``cat ShapeNetViPC-Dataset.tar.gz* | tar zx``, you will get ``ShapeNetViPC-Dataset`` contains three floders: ``ShapeNetViPC-Partial``, ``ShapeNetViPC-GT`` and ``ShapeNetViPC-View``. 

For each object, the dataset include partial point cloud (``ShapeNetViPC-Patial``), complete point cloud (``ShapeNetViPC-GT``) and corresponding images (``ShapeNetViPC-View``) from 24 different views. You can find the detail of 24 cameras view in ``/ShapeNetViPC-View/category/object_name/rendering/rendering_metadata.txt``.

Use the code in  ``utils/ViPCdataloader.py`` to load the dataset.

### Pre-trained models
Download the pre-trained models at [Google drive](https://drive.google.com/file/d/1AU2ddmVjbbdEWr5-3w2jqXt9d0nky8ts/view?usp=sharing)

### Evaluation
```
# Specify the checkpoint path in config_vipc.py
__C.CONST.WEIGHTS = "path to your checkpoint"

python main.py --test
```

### Training
```
python main.py
```

## Acknowledgement
The repository is based on [SVDFormer](https://github.com/czvvd/SVDFormer_PointSea), some of the code is borrowed from:
- [Meta-Transformer](https://github.com/invictus717/MetaTransformer)
- [ViPC](https://github.com/Hydrogenion/ViPC)
- [XMFnet](https://github.com/diegovalsesia/XMFnet)

Thanks for their opensourceing.

The point clouds are visualized with blender.

## License

This project is open sourced under MIT license.



## 条件特征流匹配版本

当前 `EGIInet` 使用以下路径：图像 tokens 与点云几何 tokens → 带模态标识的条件融合 C → 条件速度场 → Euler 积分生成缺失 latent set → 集合解码器 → 与输入点云拼接。
图像与点云共同作为流匹配的条件；被噪声化、积分生成的是缺失区域特征，而不是分别对两种条件输入加噪声。

训练时，从完整 GT 中选取距输入点云最近邻距离最大的点作为**近似**缺失区域（数据集没有显式缺失区域标签）。学习查询集合编码器将其编码为固定顺序的 latent slots。
采用直线路径 `Zt=(1-t)Z0+tZ1`，速度监督为 `Z1-Z0`；流匹配目标 latent 停止梯度，通过独立的自编码重建损失训练目标编码器，避免仅靠速度回归导致目标特征坍缩。
总损失包括完整点云 Chamfer、缺失点云 Chamfer、目标自编码 Chamfer、速度 MSE 和原有 style loss。训练中的生成路径同样从纯噪声积分，不使用 GT 作为生成条件。

在 `config_vipc.py` 中设置数据目录、输出目录、列表路径以及 `NETWORK.FLOW`：

- 默认输入采样 1024 点、生成 1024 点，输出 2048 点；输入点保持原坐标。
- `LATENT_TOKENS=64`、`MISSING_POINTS=1024`；生成点数必须是 latent token 数的整数倍。
- `TRAIN_STEPS=4`、`SAMPLE_STEPS=16`，均使用 Euler 积分；增加训练步数会增加显存和耗时。
- `TRAIN.RESUME=None` 从头训练；恢复训练时指定本版本 checkpoint。旧版权重不能直接用于新架构评估。测试需将 `CONST.WEIGHTS` 改成本版本训练得到的 checkpoint。
- `TRAIN.FLOW_WEIGHT`、`MISSING_WEIGHT`、`AUTOENCODER_WEIGHT`、`STYLE_WEIGHT` 控制损失权重，默认值未经训练调优。

训练和评估命令仍为 `python main.py` 和 `python main.py --test`。需要原项目的 CUDA 扩展和依赖。
模型默认仍返回 `(complete, style_loss)`；训练使用 `model(partial, image, gt=gt, return_details=True)` 获取各辅助输出。
直接调用模型时可传入形状为 `[B, LATENT_TOKENS, embed_dim]` 的固定 `noise`，以复现同一噪声下的推理结果；验证默认使用随机噪声。

流匹配模块测试：`python -m unittest discover -s tests -v`。这些测试只依赖 PyTorch，不依赖 CUDA 扩展；完整网络仍需 CUDA 环境验证。
