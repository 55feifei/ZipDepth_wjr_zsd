# ZipDepth Android 演示

本应用把最终选中的 `output/experiments/single/best.pth` 导出的 ONNX 完整放进 APK，在手机上离线分析后置摄像头。时序平滑是应用里的可选后处理，不属于模型权重。界面同时显示 RGB 和相对逆深度：数值越大表示越近，色标红近蓝远。输出没有米制标定，不能读成“距离多少米”。

首次安装且没有保存过尺寸时，默认使用 **384×384**。之后恢复上次手动选择的 384 或 256。已有安装如果保存过 256，会继续使用 256；验收默认演示时请手动切回 384。应用不会根据 FPS 自动切换模型。15 FPS 是本项目自己的性能目标，不是课程 PDF 规定的硬性门槛。

## 命令行构建

工程没有 `gradlew.bat` / Gradle Wrapper。构建脚本直接调用项目内的 Gradle 8.9：`output/tools/android/gradle-8.9/bin/gradle.bat`。JDK 17、Android SDK 35 和依赖缓存也在 `output/tools/android/`，不修改系统安装或 PATH。仓库路径含中文时，脚本会创建 `%LOCALAPPDATA%\ZipDepthRepo` 目录联接后再构建，避免 Windows 上的 Gradle 测试进程找不到主类。

在仓库根目录打开 PowerShell：

```powershell
Set-Location -LiteralPath 'F:\鲜可机器智能课设\ZipDepth'
powershell -ExecutionPolicy Bypass -File .\android\setup-build.ps1
powershell -ExecutionPolicy Bypass -File .\android\build.ps1 -Model selected
```

`setup-build.ps1` 只在 JDK、Gradle 或 SDK 组件缺失时需要执行。当前机器已经具备 JDK 17、Gradle 8.9、compileSdk 35 和 Build Tools 35.0.0 时，直接构建即可。依赖缓存齐全后可加 `-Offline`。

默认 `-Model selected` 使 `MODEL_FLAVOR` 为 `zipdepth_selected`，并打包实验 5600 选出的同一 checkpoint：

| 源文件 | APK 内资产 |
|---|---|
| `output/experiments_5600/models/zipdepth_selected_384.onnx` | `assets/zipdepth_384.onnx` |
| `output/experiments_5600/models/zipdepth_selected_256.onnx` | `assets/zipdepth_256.onnx` |
| `tests/fixtures/turbo_lut.json` | `assets/turbo_lut.json` |

`-Model final` 与 `-Model baseline` 仍从 `output/models/zipdepth_<model>_{384,256}.onnx` 打包。

构建前会核对两个 ONNX 与各自 JSON 里的 `onnx_sha256`，并确认它们来自同一个 checkpoint。文件缺失、哈希不一致或 APK 里的模型字节与源文件不同时，构建失败。ONNX 以不压缩方式打进 APK。成功后的产物：

```text
output/apk/zipdepth-selected-debug.apk
output/apk/build_evidence_selected.json
output/apk/apk_asset_check_selected.json
```

课程演示使用 debug APK，便于之后用 `run-as` 读取应用内部测量文件。

## Android Studio

Android Studio 不是必需的。若要导入，打开本仓库的 `android` 文件夹，选择 JDK 17，例如 `output/tools/android/jdk-17*`。`local.properties` 指向项目内 SDK。不要期待仓库里存在 Gradle Wrapper；命令行构建以上面的 `android/build.ps1` 为准。手机架构为 arm64-v8a，最低 Android 9（API 28）。

## 安装到荣耀 100 Pro

1. 手机开启开发者选项和 USB 调试，用可传输数据的线连接电脑，并在手机上允许这台电脑调试。
2. 确认设备状态为 `device`：

```powershell
& .\output\tools\android\sdk\platform-tools\adb.exe devices -l
```

`unauthorized` 需要在手机上确认授权。多台设备时，后面的命令加上 `--serial <设备号>`。

3. 安装并启动：

```powershell
& 'F:\ANACONDA\envs\ML1\python.exe' -B .\scripts\verify_android.py `
  --install `
  --apk .\output\apk\zipdepth-selected-debug.apk
```

也可以把 APK 拷到手机后从文件管理器安装。签名冲突时，卸载会清掉应用内测量数据；先按下面的方式导出或 `adb pull` 再卸载。

4. 打开「ZipDepth 实时深度」，授予摄像头权限。界面应显示 `zipdepth_selected`、当前边长、后端和模型 SHA256 前缀。

## 界面与输出含义

- RGB 预览和深度图使用同一块中心正方形视野。预处理是校正方向、中心裁剪、双线性缩放，再变成 RGB `[0,1]` 的 CHW 输入。ImageNet mean/std 已在模型内部，调用端不再标准化。
- 深度图是相对逆深度响应。高值较近，显示为红色；低值较远，显示为蓝色。
- 深度更新 FPS 统计的是深度图实际显示次数，不是摄像头预览帧率。
- 处理至显示延迟从 CameraX 分析回调进入开始，到 `ImageView.setImageBitmap` 为止，不含传感器曝光和屏幕扫描。预处理、推理、后处理三段时间可以重叠；界面上的三段毫秒数是各自耗时。`ui_queue_ms` 是总延迟减去这三段后的剩余，包含阶段之间的等待。
- 时序平滑默认开启。关闭后只做当帧着色。切换尺寸、平滑开关或回到前台都会清空平滑历史。
- 模型行里的后端来自实际会话。XNNPACK 是 CPU 加速执行提供程序，不是手机 NPU。初始化或推理失败时会重建 CPU 会话，并在测量结果里保留回退原因和最终后端。
- 电池温度只用于观察趋势，不是芯片温度。

## 景深 Bokeh 模式

打开「景深 Bokeh」开关后，下方画面从深度色图变成用当前深度渲染的浅景深预览，点击画面任意位置即可对焦到该处。实现在 `BokehRenderer.kt`，镜像 `zipdepth/bokeh.py`（BokehMe 的经典散射渲染器）：

- 相对逆深度按 2%/98% 分位归一化为视差，分位范围以 0.1 系数 EMA 更新；对焦视差取点击处 9×9 窗口中位数，以 0.3 系数平滑。新的点击或场景切换会立即跳到新焦点。
- 在半分辨率（384 时为 192×192）做散射：每个像素以模糊半径 `0.03·边长·(d − d_focus)` 散射到圆盘上，gamma 4 线性化后再还原，然后双线性放大。模糊半径小于半像素的像素直接保留原图，对焦主体不会被降采样变糊。
- 输入使用送给模型的同一张 RGB 正方形，后处理线程渲染，位图复用 `DepthColorizer` 的缓冲池。界面上的“后处理”毫秒数包含景深渲染耗时。
- 景深模式下不能开始五分钟测速；测速期间景深开关也锁定。

限制：只有经典渲染器，没有 BokehMe 的神经网络修复，前景边缘可能有轻微光晕；深度边界预测错误会直接表现成虚化错误。手机上的实际帧率尚未实测。

## 五分钟实测

点「测试 5 分钟」后，先用 10 帧采样 ONNX Runtime 后端，再额外预热 10 帧，然后记录 300 秒。测速期间尺寸和平滑开关固定。测速时不要录屏。建议四组都在同一部手机、相近室内场景和相同亮度下完成，并关闭省电模式：

| 测试 | 输入 | 时序平滑 |
|---|---:|---|
| A | 384 | 开 |
| B | 384 | 关 |
| C | 256 | 开 |
| D | 256 | 关 |

组间尽量让电池温度回到第一组开始时的约 ±2℃。进入后台、旋转导致重建或推理失败会结束本次测试，并写成未完成。

结束后点「导出测量」，或在电脑上采集。采集不会删除手机上的原始文件在电脑侧的副本：

```powershell
& 'F:\ANACONDA\envs\ML1\python.exe' -B .\scripts\verify_android.py --collect
```

每次测试目录包含：

- `frames.csv`：逐帧预处理、推理、后处理、UI 队列、处理至显示、累计更新 FPS、后端、尺寸、平滑状态和电池温度。
- `summary.json`：设备、模型 SHA256、后端、`backend_at_start`、回退原因、P50/P95、实测 FPS 和完成状态。
- `ort_provider_sample_*.json`：算子 provider 分布。`XNNPACK + CPU` 表示启用了 XNNPACK，部分算子仍可能落在 CPU。

汇总写在 `output/device/acceptance.json`。384 开、384 关、256 开、256 关按配置分开，不会互相覆盖。没有任何配置达到 15 FPS 时，`recommended_final_size` 为 null，不会把 256 标成达标推荐。

人工功能验收另外记录在 `output/device/manual_acceptance.json`。只有 `complete` 为 true，且下列检查项全部为 true，才算人工验收完成：

`first_launch_384`、`camera_granted`、`camera_denied_retry`、`camera_permanent_deny_settings`、`switch_384_256`、`smoothing_toggle`、`rotation`、`background_foreground`、`benchmark_background_incomplete`、`offline_inference`、`export_measurements`、`five_minute_stable`。

## 演示录像

四组测速结束后，让应用留在前台，再录 120 秒。录制期间展示近远物体、缓慢移动、384/256 切换和平滑开关：

```powershell
& 'F:\ANACONDA\envs\ML1\python.exe' -B .\scripts\verify_android.py `
  --record `
  --record-seconds 120
```

视频保存为 `output/device/demo.mp4`。总体完成状态只有在四组有效测量、人工验收和这段录像都存在时才为真。

## 真机结果

截至本次构建，电脑上的 `adb devices` 没有已授权手机。因此还没有安装结果、功能验收、四组五分钟数据和演示录像。这些不能用电脑推理速度代替。手机连接并完成上面的步骤后，结果应出现在：

```text
output/device/device.json
output/device/installation.json
output/device/files/measurements/
output/device/acceptance.json
output/device/manual_acceptance.json
output/device/demo.mp4
```

## 故障排查

- 模型载入失败：确认安装的是 `zipdepth-selected-debug.apk`，而不是缺少 ONNX 的中间包。构建日志里的哈希必须和 `output/experiments_5600/models/zipdepth_selected_*.json` 一致。
- 摄像头权限被拒绝：点「重试」。若系统不再弹出权限框，重试会打开应用设置，授权后返回应用再点重试。
- 有预览但没有深度：看模型行是否仍停在“载入”。推理失败时统计行会给出原因，点「重试」重建会话。
- 深度图方向不对：预览和深度共用 1:1 视口；旋转后应重新对齐。若仍不对，不要用截图代替实机旋转检查。
- 安装提示签名冲突：这是调试签名变化。先导出已有测量，再卸载旧包后安装。
- 关闭 Wi-Fi 和移动数据后仍应能加载模型并推理。应用清单没有网络权限，模型在 APK 内。
- 构建找不到 `gradlew.bat` 是预期情况。使用 `android/build.ps1`，不要临时生成一套不同版本的 Wrapper。

## 实现与验证

摄像头使用 CameraX 最新帧策略。预处理、推理、后处理在三条单线程上流水执行，阶段之间只保留最新一帧。每个 `ImageProxy` 都在预处理结束后于 `finally` 中关闭。深度图位图复用，未显示或已被替换的位图回到缓冲池。模型加载、推理和会话关闭只在推理线程上。切换尺寸会关闭旧会话并丢弃过期结果。

Kotlin 单元测试覆盖共享平滑黄金序列、景深渲染与对焦跟踪的共享黄金数据（`tests/fixtures/bokeh.json`）、重置、非有限输入、YUV stride、旋转映射，以及未保存尺寸时默认 384。Python 测试覆盖模型哈希不匹配、APK 资产被压缩或字节不一致，以及四组测量按尺寸和平滑状态分开汇总。这些都不能代替真机上的权限、预览、后端分布和连续测速。
