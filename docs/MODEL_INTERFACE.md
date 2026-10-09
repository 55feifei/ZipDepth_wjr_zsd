# 电脑与 Android 共用接口

模型：ZipDepth base / balanced / upsample_unfold=False。
导出使用未融合训练权重的独立 eval 副本。训练期间不融合，不改写模型注意力。

- 输入名 `image`：float32 `[1,3,S,S]`，S=256 或 384，RGB、范围 `[0,1]`。
- RGB 先按旋转后的尺寸中心裁剪正方形，再双线性缩放；Android展示同一裁剪视野。
- 模型已内置 ImageNet mean/std，调用端不得重复标准化。
- 输出名 `depth`：float32 `[1,1,S,S]`，非负相对逆深度响应；不能标注为米。
- 所有基础模型参数严格加载；仅去掉 module./_orig_mod. 前缀。
- 高响应对应近处，Turbo 色图高值红、低值蓝。导出元数据保存规格与 SHA256。

## 在线平滑

Android `TemporalSmoother` 与共享夹具使用平面 CHW：RGB 为 float32 `[0,1]`，下标 `c*H*W + y*W + x`；深度为 HW float32。Python `GatedEMA.update` 的参数仍是 HWC，测试在调用前把夹具从 CHW 转置。两端门控参数相同。
首帧直接返回。若当前RGB与历史RGB的全图通道绝对差均值大于0.25，重置并直接返回。
否则逐像素历史权重 `w=0.3*exp(-mean_channel(abs(rgb-current_history_rgb))/0.08)`。
深度边缘：当前深度的向左和向上绝对差之和，除以 `max(P98-P2,1e-6)`，大于0.1的像素将w设为0。
第一行向上差为0，第一列向左差为0；百分位采用线性插值（NumPy默认语义）。
输出 `(1-w)*current_depth+w*previous_smoothed_depth`，更新历史RGB和输出。
参数变更、摄像头/分辨率变更、恢复前台或显式reset均清空历史。非有限输入必须拒绝。

在线色标采用P2/P98，按0.05系数更新上下界，场景切换重置；离线对比视频所有方法使用相同固定色标。
验证夹具 `tests/fixtures/smoothing.json` 包含连续帧RGB、深度及Python预期输出。
