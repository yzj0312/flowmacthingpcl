# 图像引导的缺失区域潜空间流匹配点云补全

本项目输入一张 RGB 图像和对应物体的残缺点云，通过条件流匹配生成缺失区域的潜变量，再解码为缺失点云，与已观测点云拼接得到补全结果。

当前基线采用 **DINOv2 图像编码器 + PointNet++ 点云编码器 + 缺失区域目标编码器 + 条件 Transformer 速度网络 + 可学习查询解码器**。图像与点云分别编码，暂不共享编码器权重。

## 1. 整体框架

训练阶段包含条件分支和目标分支：

```text
图像 ─────→ DINOv2 ────────→ 图像 token ───┐
                                          ├→ 条件 C
残缺点云 ─→ PointNet++ ────→ 几何 token ───┘
                                              ↓
完整 GT + 残缺点云 → 近似缺失区域 → 目标编码器 → Z₁
                                              ↓
标准高斯 Z₀ + 随机时间 t + Z₁ → Zₜ → 速度网络(Zₜ,t,C)
                                              ↓
                                    与 Z₁−Z₀ 计算 MSE
```

推理阶段无需完整 GT 或目标编码器：

```text
图像 + 残缺点云 → 条件 C
标准高斯 Z₀ → Euler 多步积分 → 预测缺失 latent
                               ↓
                         缺失区域解码器
                               ↓
                         预测缺失点云
                               ↓
                   与原始观测点云拼接 → 补全点云
```

**64 是 latent token 数，每个 token 默认 256 维，不是 64 个三维点。** 目标缺失区域和解码结果默认均为 1024 点。

## 2. 各模块设计

| 模块 | 当前实现 | 默认输出 |
| --- | --- | --- |
| 图像条件编码器 | 预训练 DINOv2 ViT-S/14，保留 patch token，再线性投影和归一化 | `[B,256,256]`，输入为 224×224 图像 |
| 点云条件编码器 | PointNet++ SSG，两级 FPS、半径邻域、局部 MLP、最大池化；位置编码及两层 Transformer | `[B,128,256]`，输入点数至少 128 |
| 条件组织 | 点云和图像 token 加模态标识后拼接 | `C: [B,384,256]` |
| 目标编码器 | 独立 PointNet++，默认采样至 256/128 个中心；64 个 learned queries 交叉注意力聚合，再经两层 latent 自注意力 | `Z₁: [B,64,256]` |
| 速度网络 | 4 层 Transformer decoder block，8 个注意力头；latent 槽位编码、正弦时间编码和 MLP | `[B,64,256]` |
| 缺失区域解码器 | 1024 个 learned queries 读取 latent，LayerNorm + MLP 输出 XYZ | `[B,1024,3]` |

条件点云编码器与目标编码器结构相似，**权重独立**。速度网络以当前 latent 为 query，以条件 C 为 key/value。解码器只接收 latent；观测点云在解码完成后单独拼接。

PointNet++ 使用纯 PyTorch 实现，不依赖原项目的自定义 CUDA 算子。默认邻域点数为 32，半径为 0.1/0.2，半径需结合数据坐标尺度检查。部分模块参数可通过 Python 构造函数调整；当前命令行采用默认网络结构。

## 3. 近似缺失区域构造

对每个完整 GT 点，计算其到残缺点云的最近邻距离：

```text
r(g) = min_{p ∈ P} ||g − p||₂
M = {g ∈ GT : r(g) > threshold}
```

- 默认 `threshold=0.03`，单位与输入坐标一致，可通过 `--threshold` 修改。
- 候选点达到 1024 个时，使用 FPS 采样为 1024 点；不足时重复候选点至 1024 点。
- 候选区域为空时直接报错，需要检查阈值、坐标对齐或筛除该样本。

未提供 `missing` 时，Dataset 在读取当前样本时按需构造，不在训练入口对全量数据运行 `torch.cdist`。提取函数同时对 GT 和观测点分块，默认每块最多 `512×512`，float32 距离输出约 1 MiB（不代表进程总内存）。可通过 `--missing-chunk-size 256` 进一步减小距离块。

按需提取不缓存结果，每个 epoch 会重新计算；长时间训练建议预先逐样本生成 `missing` 并保存，避免重复开销。提供 `missing` 后会直接读取，不再提取。当前 `.pt` 文件仍由 `torch.load` 整体读入 CPU 内存，本次修改限制的是额外的距离矩阵；大型图像/点云文件仍需有足够内存，后续可采用分片数据集。

这是近似缺失区域伪标签，不是真实可见性标签。GT 和残缺点云必须处于相同坐标系，采用相同尺度变换；不能分别重新居中或归一化。正式训练前应可视化检查缺失区域及其边界。1024 是固定输入规模，不表示每个样本实际都缺失相同数量的点。

## 4. 两阶段训练

### 阶段 A：缺失区域自编码器预训练

```text
缺失区域 M → TargetEncoder → Z₁ → MissingRegionDecoder → 重建缺失区域
```

联合训练目标编码器与解码器，损失为对称平方 Chamfer Distance（CD-L2）：两方向最近邻平方距离均值之和，再除以 2。

先检查真实缺失区域的重建效果，再进入阶段 B。目标 latent 必须保留可解码的几何信息，仅有较低流匹配 MSE 不足以证明补全有效。

### 阶段 B：条件潜空间流匹配

每个训练样本独立采样标准高斯 latent 与时间：

```text
Z₀ ~ Normal(0,I)              # [B,64,256]
t  ~ Uniform(0,1)             # 每个样本一个时间
Z₁ = E_target(M)
Zₜ = (1−t)Z₀ + tZ₁
目标速度 = Z₁−Z₀
L_FM = mean((vθ(Zₜ,t,C) − (Z₁−Z₀))²)
```

| 模块 | 阶段 A | 阶段 B |
| --- | --- | --- |
| 目标编码器 | 训练 | 冻结，BatchNorm 保持 eval |
| 缺失区域解码器 | 训练 | 冻结 |
| DINOv2 主干 | 不使用 | 默认冻结并保持 eval |
| 图像投影层、条件 PointNet++、模态标识 | 不使用 | 训练 |
| 速度网络 | 不使用 | 训练 |

阶段 B 只使用速度 MSE，不使用原项目风格损失、额外几何损失、latent 标准化或 classifier-free guidance。缺失区域仅提供监督终点，不输入条件分支。

## 5. 数据格式与环境

当前入口读取 `.pt` 张量字典，尚未直接接入原 ViPC 数据加载器。`S` 表示样本数，`B` 表示 batch 大小。

| 字段 | 张量形状 | 使用阶段 |
| --- | --- | --- |
| `missing` | `[S,1024,3]` | 自编码器训练、流匹配训练 |
| `partial` | `[S,N,3]` | 流匹配训练、推理；自动提取缺失区域时也需要 |
| `image` | `[S,3,224,224]` | 流匹配训练、推理 |
| `gt` | `[S,G,3]` | 未提供 `missing` 时用于自动构造 |

输入使用 CPU float32 张量保存，同一数据文件内样本数一致，点云点数应能直接堆叠。图像是 `[0,1]` 范围的 RGB，ImageNet 归一化由图像编码器内部完成，外部不要重复归一化。图像宽高必须是 14 的倍数。

依赖包括 PyTorch 及 DINOv2 官方仓库所需依赖。代码使用 `torch.load(..., weights_only=True)`，需使用支持该参数的 PyTorch 版本。当前仓库尚未提供经过完整训练验证的依赖锁定版本或训练权重。

首次构造流模型可能通过 Torch Hub 下载 DINOv2 代码和预训练权重，即使随后加载流匹配检查点也会先构造主干。离线使用可通过 Python API 创建 `MultimodalConditionEncoder`，设置 `dino_source='local'`、`dino_repo` 为本地 DINOv2 仓库，并传入官方 ViT-S/14 原始主干权重路径 `dino_weights`，随后传给 `LatentCompletion`。这些离线选项目前未暴露为命令行参数。

## 6. 运行方式

### 自编码器预训练

```bash
python main.py --stage ae --data train.pt --output checkpoints/ae.pt --epochs 100 --batch-size 8
```

数据提供 `missing` 即可；若不提供，则必须提供 `gt` 和 `partial` 进行近似缺失区域构造。

### 流匹配训练

```bash
python main.py --stage flow --data train.pt --ae-checkpoint checkpoints/ae.pt --output checkpoints/flow.pt --epochs 100 --batch-size 8
```

应使用已经检查过重建质量的自编码器检查点。默认优化器为 AdamW，学习率 `1e-4`；可使用 `--lr` 修改。

### 恢复训练

```bash
python main.py --stage flow --data train.pt --checkpoint checkpoints/flow.pt --output checkpoints/flow.pt --epochs 200
```

`--epochs` 为目标总轮数，不是额外轮数。只能恢复相同阶段的检查点。每轮保存模型、优化器、阶段和轮数；没有保存随机状态，因此不保证逐步复现中断前的随机序列。恢复优化器时沿用检查点中的参数组设置。

### 采样与补全

```bash
python main.py --stage sample --data test.pt --checkpoint checkpoints/flow.pt --output predictions.pt --steps 32
```

`test.pt` 只需 `image` 和 `partial`。默认从高斯 latent 开始，以 32 步 Euler 方法积分；条件 C 在积分前计算一次并复用。推理结果保存为：

| 输出字段 | 默认形状 | 含义 |
| --- | --- | --- |
| `latent` | `[S,64,256]` | 生成的缺失区域潜变量 |
| `missing` | `[S,1024,3]` | 生成的缺失点云 |
| `complete` | `[S,N+1024,3]` | 观测点云与生成点云直接拼接 |

当前不对完整输出额外去重或重采样。`--device` 可指定设备，默认优先 CUDA，否则 CPU；纯 PyTorch FPS 和完整距离矩阵在较大点云上可能较慢或占用较多内存。

## 7. 核心代码

```text
main.py                           两阶段训练、恢复和采样入口
models/
  condition_encoders.py           DINOv2、PointNet++、条件 token 组织
  target_encoder.py              缺失区域目标编码器
  missing_region_decoder.py      可学习查询缺失区域解码器
  latent_completion.py           缺失区域构造、自编码器、速度网络、流匹配及采样
run/
  stages.py                      两阶段优化器更新
tests/                           模块测试
```

历史数据工具、配置及早期实验文件可能仍在目录中；上述文件构成当前主流程。旧 `config_vipc.py` 不控制新 `main.py` 的运行参数。原论文解码器和依赖它的旧训练、测试入口已移除。

## 8. 验证状态与后续实验

目前已编写编码器、目标编码器、解码器和流匹配相关测试，覆盖输出形状、梯度、冻结行为和采样等接口。可执行：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

**目前只完成静态语法检查。当前执行环境缺少 PyTorch，运行测试未能启动；尚未完成真实 DINOv2 前向、数据集训练或补全效果评估。** 测试中的替代图像主干用于避免下载，不代表真实预训练模型已经验证。

后续重点验证：

1. 近似缺失区域是否合理，目标自编码器能否准确重建。
2. 生成缺失区域及拼接完整点云的 CD、F-score 等指标。
3. 移除或打乱图像、点云条件后，性能如何变化。
4. 16/32/64 步采样的质量与耗时。
5. 独立、共享、部分共享编码方式的对照；当前基线仅实现独立编码。

当前入口提供训练与采样，尚未包含完整验证集评估、最佳模型选择和基准报告流程。

## 9. 来源与许可

本仓库由原点云补全项目逐步改造，当前主框架已调整为缺失区域潜空间条件流匹配。保留原仓库 `LICENSE` 及已有来源声明；使用预训练模型时同时遵循其许可。

- 原项目论文：[Explicitly Guided Information Interaction Network for Cross-modal Point Cloud Completion](https://arxiv.org/abs/2407.02887)。
- 图像主干：[DINOv2 官方仓库](https://github.com/facebookresearch/dinov2)。
- 点云局部编码基础：[PointNet++](https://arxiv.org/abs/1706.02413)。
- 流匹配方法背景：[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)。
