# ZipDepth Android 演示

本工程将经过一致性验证的 ZipDepth ONNX 完整打包到 APK，在手机上离线分析后置摄像头。界面显示 RGB 与相对逆深度，数值越大表示越近，色标为红近蓝远。输出没有米制标定。

## 命令行构建

在仓库根目录打开 PowerShell，运行：

```powershell
powershell -ExecutionPolicy Bypass -File android/setup-build.ps1
powershell -ExecutionPolicy Bypass -File android/build.ps1 -Model baseline
```

安装器只使用官方 Adoptium、Gradle 和 Google 下载地址，检查工具包校验值。JDK 17、Gradle 8.9、SDK 35 与缓存保存在 `output/tools/android/`，不修改系统安装或 PATH。第一次需要联网下载工具、Android SDK 和 Maven 依赖；依赖缓存齐全后可加 `-Offline` 构建。

构建前需存在 `output/models/zipdepth_baseline_256.onnx` 和 `zipdepth_baseline_384.onnx`，以及共享 `tests/fixtures/turbo_lut.json`。Gradle 自动将真实模型复制到生成资产；文件缺失时直接失败。微调模型经重新验证并以 `zipdepth_final_256.onnx`、`zipdepth_final_384.onnx` 保存后，运行 `android/build.ps1 -Model final`。生成 APK 为 `output/apk/zipdepth-baseline-debug.apk` 或 `zipdepth-final-debug.apk`，并打印 SHA256。

## Android Studio 导入

打开本仓库的 `android` 文件夹，选择 JDK 17；可选用安装脚本产生的 `output/tools/android/jdk-17...`。`local.properties` 已写入项目内 SDK 地址，Gradle Wrapper 使用 8.9。手机运行所需架构为 arm64-v8a，最低 Android 9（API 28）。

## 荣耀 100 Pro 安装

1. 在手机设置中开启开发者选项与 USB 调试，用可传输数据的 USB 线连接电脑，并在手机上允许本机调试。
2. 用 `output/tools/android/sdk/platform-tools/adb.exe devices -l` 检查设备状态为 `device`；`unauthorized` 需要在手机确认授权。
3. 执行 `adb install -r output/apk/zipdepth-baseline-debug.apk`，也可以传输 APK 后从手机文件管理器安装。
4. 打开「ZipDepth 实时深度」，授予摄像头权限。权限拒绝时可点「重试」或进入应用设置重新授权。

先用 256 输入。384 通常提供更多细节，但需以实际持续测速决定是否达到 15 FPS。界面 FPS 统计实际深度图更新次数；摄像头预览流畅不代表深度推理达到同一帧率。

## 五分钟实测与结果导出

点「测试 5 分钟」后，程序先用 10 帧生成 ONNX Runtime 后端事件采样，再额外预热 10 帧，然后记录 300 秒。测试期间固定输入尺寸和平滑开关，结束后点「导出测量」获得：

- `frames.csv`：逐帧预处理、推理、后处理、UI 队列与处理至显示时间、累计更新 FPS、后端、输入尺寸、平滑状态和电池温度。
- `summary.json`：设备系统、模型 SHA256、实际后端事件统计、P50/P95、实测 FPS 与完成状态。
- `ort_provider_sample_*.json`：真实算子 provider 分布；显示 `XNNPACK + CPU` 表示启用 XNNPACK 且可能有算子落到 CPU。

处理至显示延迟从 `ImageProxy` 分析回调开始到 `ImageView.setImageBitmap`，不包含传感器曝光和屏幕扫描。电池温度用于观察发热趋势，不等价于芯片温度。后台、旋转重建、推理错误会结束当前测试并标为未完成；返回后可重新开始。结果在应用内部 `files/measurements/`，调试 APK 也可用 `adb shell run-as edu.course.zipdepth` 读取。

## 实现与验证

摄像头使用 CameraX 最新帧策略和单个后台分析线程；所有帧在 `finally` 关闭。YUV 转 RGB 明确处理两个 stride，先旋转、中心正方形裁剪再双线性缩放，生成 RGB `[0,1]` NCHW 输入；ImageNet 标准化保留在模型内部。RGBA 文档和不同设备的缓冲区通道排列可能引入歧义，因此本版本采用明确的 YUV 平面读法。

ONNX Runtime 优先 XNNPACK，初始化或推理失败时重建 CPU 会话并记录原因。门控 EMA 与 Python 共享黄金序列；RGB 变化抑制历史权重，深度边缘不混合；切场景、恢复前台、改变尺寸或平滑状态均重置历史。显示色标以 P2/P98 缓慢更新，防止每帧 min/max 拉伸造成额外闪烁。

构建执行 Kotlin 单元测试：共享 Python 平滑黄金序列、重置、非有限值输入、YUV stride 和旋转映射。它们验证算法与部分摄像头处理数学；真正的权限、预览映射、后端分布、连续测速和系统兼容性必须在已连接手机上完成。
