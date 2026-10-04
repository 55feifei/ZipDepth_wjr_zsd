# 基于 ZipDepth 的移动端实时视频深度估计

本项目新增 TartanAir 相邻帧微调、序列时序评测、可信 ONNX 导出和 Android 摄像头演示。进度与验证证据见 [TODO.md](TODO.md)。本阶段不含 PPT 和实验报告。

## 环境与模型

使用现有 `F:\ANACONDA\envs\ML1\python.exe`，已验证 Python3.12.13、torch2.6.0+cu124、CUDA12.4、RTX4060 Laptop 8GB。不要直接整体安装原仓库 requirements.txt（它要求更高版本torch）。

```powershell
$coursePython = 'F:\ANACONDA\envs\ML1\python.exe'
& $coursePython -m pip install -r requirements-course.txt --index-url https://pypi.org/simple
& $coursePython -B -m unittest discover -s tests -v
# 可选真实GPU前后向与恢复检查；正式训练期间不要同时运行。
$env:ZIPDEPTH_GPU_SMOKE = '1'
& $coursePython -B -m unittest discover -s tests -p test_temporal_loss.py -v
Remove-Item Env:ZIPDEPTH_GPU_SMOKE
```

实际版本快照在 `output/environment/`。torchvision不是新增模块的必需依赖。ONNX采用torch2.6 legacy静态导出，不需要onnxscript。

预训练权重来自仓库随附 `checkpoints/zipdepth_base_npu.pth`。模型为base/balanced/无unfold结构；所有基础参数严格加载。训练保持未融合结构，部署只在独立eval副本上融合。NPU后缀指算子结构便于部署，不表示手机已经调用NPU。

模型输入RGB `[1,3,S,S]`，范围[0,1]，中心正方形裁剪后双线性缩放；ImageNet归一化在模型内部。输出是**相对逆深度响应，不是米制距离**。详情见 `docs/MODEL_INTERFACE.md`。

## 数据、训练与评测

[TartanAir官方工具](https://github.com/castacks/tartanair_tools)提供数据结构及官方HuggingFace镜像。仅选择三个场景Easy左目RGB、深度、光流与掩码；下载清单实际49.718GB。下载前查预算，partial续传，SHA256及ZIP CRC校验后才提取。文件存于data/，不进入源码版本管理。

```powershell
& $coursePython -B scripts/download_tartanair.py --help
& $coursePython -B scripts/prepare_tartanair.py --help
& $coursePython -B -u scripts/download_tartanair.py --workers 3 --range-workers 8
# 等待下载/校验/解压完成；完整轨迹划分防止相邻帧跨集合。
& $coursePython -B scripts/prepare_tartanair.py --root data/tartanair --output data/tartanair/manifest.json
& $coursePython -B scripts/run_course_experiments.py --manifest data/tartanair/manifest.json
```

精确下载命令、进程和状态见 `output/data_download/`。小样路径以`data/tartanair_bootstrap_manifest.json`的root字段为准，从官方远端完整ZIP中CRC校验提取；若用于训练，必须带`--bootstrap`，结果仅验证流水线，不能表述为完整课设实验。每次实验冻结索引；恢复已有实验时使用其`frozen_manifest.json`，防止下载中新场景改变训练集。

```powershell
& $coursePython -B scripts/run_course_experiments.py --manifest data/tartanair_bootstrap_manifest.json --output output/smoke --max-updates 2 --bootstrap --skip-calibration
& $coursePython -B scripts/train_temporal.py --manifest data/tartanair/manifest.json --output output/experiments/temporal_0.1 --temporal-weight 0.1
& $coursePython -B scripts/train_temporal.py --manifest data/tartanair/manifest.json --output output/experiments/temporal_0.1 --temporal-weight 0.1 --resume output/experiments/temporal_0.1/last.pth
```

训练默认384、BF16、AdamW1e-5、每批一对帧、累积4、冻结BN运行统计、关闭compile，启用确定性计算。三组λ0/0.1/0.3同初始化、同帧对与更新数，单GPU串行；先短程校准，再设置共同预算，累计12小时，其中正式训练最多9小时。

空间项使用SSI＋多尺度梯度；时序项对光流对齐的**预测误差**做一致性，扣除真值真实运动变化。无效/遮挡/越界像素排除；全空验证不视为最优。训练日志、有效配置、恢复随机状态、best/last保存在实验目录。

```powershell
& $coursePython -B scripts/evaluate_temporal.py --manifest data/tartanair/manifest.json --split val --models original=checkpoints/zipdepth_base_npu.pth single=output/experiments/single/best.pth temporal=output/experiments/temporal_0.1/best.pth
```

空间精度按每帧仿射对齐逆深度后转换为米制真值比较AbsRel/RMSE/δ1；时序精度每条完整轨迹仅拟合一次仿射，报告光流对齐误差变化和尺度/偏移漂移。最终模型仅由验证集选择，AbsRel不超过单帧对照×1.02；测试集在选型之后评估。10%时序改善是目标，不是预先保证的结果。

## 导出、视频和设备性能

```powershell
& $coursePython -B scripts/export_mobile.py
& $coursePython -B scripts/export_mobile.py --checkpoint output/experiments/temporal_0.1/best.pth --prefix zipdepth_final
& $coursePython -B scripts/demo_video.py --max-frames 90 --models baseline=output/models/zipdepth_baseline_256.onnx baseline_smooth=output/models/zipdepth_baseline_256.onnx --smooth-models baseline_smooth
& $coursePython -B scripts/benchmark_course.py --iterations 200
```

export_mobile会先检查原始PyTorch→融合PyTorch→ONNX的一致性，而不是只检查shape；metadata含权重来源、SHA256、输入规格、opset17和实际误差。默认容差atol1e-4/rtol1e-3。

视频各方法用相同固定色标。播放FPS是输入视频速率，不能当作推理FPS；侧车JSON单独记录实测耗时。在线EMA历史权重≤0.3，由RGB变化和深度边缘门控，并在场景切换时重置。Python/Kotlin共享黄金序列及精确Turbo查找表。

Android工程在 `android/`，使用专用开发工具（`output/tools/android`）、Kotlin/CameraX/ORT，arm64、最低API28；真实256/384 ONNX作为构建输入，缺模型构建失败。构建、安装和5分钟测速方法见Android运行说明。优先XNNPACK，异常重建CPU会话；后端profiling与实际深度更新FPS单独记录。手机未连接时，只有APK可用性，不能声称真机性能已验证。

```powershell
# 手机已连接并授权后安装、启动；在应用中分别运行256/384的五分钟测试。
& $coursePython -B scripts/verify_android.py --install --apk output/apk/zipdepth-baseline-debug.apk
# 正式模型完成后改用zipdepth-final-debug.apk。采集实际CSV/JSON并检查模型来源。
& $coursePython -B scripts/verify_android.py --collect
# 手机前台演示时录制120秒，输出output/device/demo.mp4。
& $coursePython -B scripts/verify_android.py --record --record-seconds 120
& $coursePython -B scripts/package_course.py
```

实际曲线与指标图由`plot_course_results.py --input output/experiments --output output/curves/formal`生成；小样使用`--input output/smoke --output output/curves/smoke`，图中明确标为SMOKE。图表来源及校验值见provenance.json。

完整数据下载期间，`scripts/course_pipeline.py`作为本地后台作业等待12个归档校验/提取完成，然后执行正式训练、评测、四组视频、最终APK和软件打包。状态见`output/pipeline/status.json`，逐阶段“做什么/为什么/选择理由”见`progress.md`。此进程需要电脑保持运行；不是定时服务。等待阶段可建立`output/pipeline/STOP`停止等待，恢复前移除该文件；训练中断可由last.pth恢复。

## 结果、限制与AI使用

- 原模型ONNX：`output/models/zipdepth_baseline_{256,384}.onnx`；基线电脑视频：`output/videos/baseline_demo.mp4`。
- 电脑性能：`output/benchmarks/pc.json`；真实微调结果以实验目录的`selection.json`为准。
- 下载小样、正式数据、原始模型和微调模型必须区分标注；不得借用论文速度充当实测。
- TartanAir是合成数据，在真实手机场景的泛化需要视频与真机检查；门控EMA仍可能出现快速运动拖影。
- AI用于辅助实现、审查、测试和运行说明。关键算法已建立行为与数值验证；使用者需理解尺度不确定性、时序误差定义和部署前后处理，现场能够解释。
- 未提供成员姓名和实际分工，本阶段不填署名或贡献；不代发邮件、不填写外部表格。
