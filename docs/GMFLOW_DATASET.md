# dataset.py 与 GMFlow 光流估计

## 老师提供的数据集

本次指定的输入是 `data/tartanair/TartanAir`，包含 24 条轨迹、5,583 对相邻帧，图像大小为 480×640。对应输出目录是 `data/tartanair/TartanAir_flow`，保留原数据中的多层场景与轨迹结构。

```powershell
$coursePython = 'F:\ANACONDA\envs\ML1\python.exe'
& $coursePython -B -u scripts/estimate_flow.py --input data/tartanair/TartanAir --device cuda
& $coursePython -B dataset.py --data-dir data/tartanair/TartanAir --mode val --seq-len 12 --crop-size 384
```

本次使用本地 `gmflow/gmflow/pretrained/gmflow_things-e9887eda.pth`，在原始分辨率进行双向推理。生成日志及输入清单位于 `output/gmflow_teacher/`。验证脚本 `output/gmflow_teacher/validate_teacher_flow.py` 会逐个检查全部生成文件，并实际读取每条轨迹的 12 帧验证序列；完成后的报告保存为 `output/gmflow_teacher/validation.json`。

## 原有样本与生成命令

在仓库根目录使用已有 ML1 环境运行：

```powershell
$coursePython = 'F:\ANACONDA\envs\ML1\python.exe'

# 生成相邻帧双向光流，默认加载本地 Things 预训练权重。
& $coursePython -B scripts/estimate_flow.py --input data/tartanair_bootstrap --device cuda

# 查看完整数据集的轨迹和帧对数量。
& $coursePython -B scripts/estimate_flow.py --input data/tartanair --dry-run

# 生成完整数据；重跑时跳过四个输出文件都已存在的帧对。
& $coursePython -B scripts/estimate_flow.py --input data/tartanair --device cuda
```

短样本集 `data/tartanair_bootstrap` 包含 9 条轨迹、36 个相邻帧对。本次已使用真实 GMFlow checkpoint 和 CUDA 完成全部短样本集推理，结果位于 `data/tartanair_bootstrap_flow`。完整 `data/tartanair` 的光流需要运行上面的完整生成命令。

可以指定 `--output` 自定义输出位置、`--max-pairs 10` 限制本次新增帧对数、`--overwrite` 重新生成。`--inference-size 384 512` 可降低推理分辨率，输出仍恢复为原图尺寸和原图像素单位；该选项会改变估计精度。没有 CUDA 时可用 `--device cpu`。

每条轨迹输出四个目录：

```text
tartanair_bootstrap_flow/<场景>/Easy/<轨迹>/
  f_flow/000000_left.npy
  b_flow/000000_left.npy
  f_mask/000000_left.npy
  b_mask/000000_left.npy
```

四个文件均使用帧对第一帧的文件 stem 命名。`f_flow` 表示第 t 帧到第 t+1 帧，`b_flow` 表示第 t+1 帧到第 t 帧，数组为 `[H,W,2]` float32，两个分量分别为水平、垂直像素位移。磁盘 mask 为 `[H,W]` uint8，`0` 有效、`1` 无效；无效像素包含前后向不一致和运动终点越界。

读取生成结果：

```python
from dataset import DepthVideoDataset
from torch.utils.data import DataLoader

dataset = DepthVideoDataset(
    'val', data_dirs='data/tartanair_bootstrap', crop_size=384, seq_len=4,
)
sample = dataset[0]
batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
```

命令行读取验证：

```powershell
& $coursePython -B dataset.py --data-dir data/tartanair_bootstrap --mode val --seq-len 4 --crop-size 384
```

数据集默认查找输入根目录同级的 `<输入目录名>_flow`。也支持轨迹内部的 `f_flow/b_flow/f_mask/b_mask`。如果使用自定义 `--output`，需将生成的四个目录放在相应轨迹内，或保持上述默认同级目录结构。

| 样本键 | 单个样本形状 | 含义 |
| --- | --- | --- |
| `image` | `[T,3,H,W]` | RGB float32，范围 `[-1,1]` |
| `depth` | `[T,1,H,W]` | 逆深度，非米制深度 |
| `f_flows`, `b_flows` | `[T-1,2,H,W]` | 经过缩放、裁剪后的双向像素位移 |
| `f_masks`, `b_masks` | `[T-1,1,H,W]` | **返回值**为 `1` 有效、`0` 无效 |

训练模式同步缩放、裁剪和水平翻转所有帧；水平翻转只改变光流水平分量的符号。验证模式使用确定性中心裁剪。同一轨迹最后一个完整序列留给验证，训练使用之前的完整序列；仅有一个完整序列的轨迹没有训练样本。这一接口沿用课设模板的划分方式。仓库原有 `TartanAirPairDataset` 和 manifest 训练流程仍使用自身的轨迹划分与官方光流。

也可以在已经读取的样本上重新估计光流：

```python
from zipdepth.optical_flow import GMFlowEstimator

estimator = GMFlowEstimator(device='auto')
sample.update(estimator.infer_dataset_sample(sample))
```

`infer_pair(rgb0, rgb1)` 接受 NumPy `[H,W,3]` 或 torch `[3,H,W]` RGB，数值范围为 `[0,255]`，返回与磁盘文件相同语义的四个数组。模型实例可重复用于多个帧对。

本次联合验证读取了全部 9 条四帧验证序列和 9 条两帧训练序列，并检查批量加载、张量形状、数值有限性、二值掩码和验证集重复读取一致性。验证记录保存在 `output/gmflow_dataset_validation.json`。数据集专用测试 9 项、光流接口测试 2 项、原有 TartanAir 回归测试 7 项均通过。
