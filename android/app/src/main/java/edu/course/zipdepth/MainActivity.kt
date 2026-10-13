package edu.course.zipdepth

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.provider.Settings
import android.util.Rational
import android.util.Size
import android.view.MotionEvent
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import edu.course.zipdepth.databinding.ActivityMainBinding
import java.io.File
import java.util.ArrayDeque
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.RejectedExecutionException
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.min

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    @Volatile private var inferenceThread: Thread? = null
    private val cameraExecutor = Executors.newSingleThreadExecutor { worker -> Thread(worker, "zipdepth-camera") }
    private val inferenceExecutor = Executors.newSingleThreadExecutor { worker ->
        Thread({
            inferenceThread = Thread.currentThread()
            worker.run()
        }, "zipdepth-infer")
    }
    private val postExecutor = Executors.newSingleThreadExecutor { worker -> Thread(worker, "zipdepth-post") }
    private val inferHandoff = LatestHandoff()
    private val postHandoff = LatestHandoff()
    @Volatile private var engine: DepthEngine? = null
    @Volatile private var preprocessor: YuvPreprocessor? = null
    @Volatile private var slotPool: SlotPool? = null
    private var colorizer: DepthColorizer? = null
    private var bokehRenderer: BokehRenderer? = null
    private var bokehPixels = IntArray(0)
    private var appliedFocusVersion = -1
    @Volatile private var bokehEnabled = false
    @Volatile private var focusPoint = FocusPoint(.5, .5, 0)
    private var unsmoothedPreviousRgb: FloatArray? = null
    private val smoother = TemporalSmoother()
    private lateinit var recorder: BenchmarkRecorder
    private var cameraProvider: ProcessCameraProvider? = null
    private var analysis: ImageAnalysis? = null
    private var cameraBindingPending = false
    @Volatile private var resumed = false
    @Volatile private var generation = 0
    @Volatile private var smoothingEnabled = true
    @Volatile private var measurementPending = false
    @Volatile private var processingFailed = false
    @Volatile private var measurementHint = ""
    @Volatile private var profileWarmup = 0
    @Volatile private var measureWarmup = 0
    @Volatile private var recordFrames = false
    @Volatile private var recordingStartNs = 0L
    private var profiling = false
    private var providerProfile: File? = null
    private var measurementFiles: List<File> = emptyList()
    private val benchmarkOpen = AtomicBoolean(false)
    private val displayedTimes = ArrayDeque<Long>()
    private val pendingDisplay = AtomicReference<PendingDisplay?>(null)
    private var displayedBitmap: Bitmap? = null
    private var permissionAsked = false

    private data class PendingDisplay(
        val bitmap: Bitmap,
        val token: Int,
        val startedNs: Long,
        val timing: FrameTiming,
        val label: String,
        val engine: DepthEngine,
        val smoothing: Boolean,
        val recordable: Boolean,
    )

    /** Focus point in [0,1] of the square model frame; a new version snaps the focus instead of easing. */
    private data class FocusPoint(val u: Double, val v: Double, val version: Int)

    private val permissionRequest = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else binding.stats.text = "需要摄像头权限。点击重试授予权限或打开应用设置。"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_YES)
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.root.keepScreenOn=true
        val padding = (12 * resources.displayMetrics.density).toInt()
        ViewCompat.setOnApplyWindowInsetsListener(binding.root) { view, insets ->
            val bars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(padding+bars.left,padding+bars.top,padding+bars.right,padding+bars.bottom)
            insets
        }
        recorder = BenchmarkRecorder(this)
        permissionAsked = savedInstanceState?.getBoolean("permissionAsked") ?: false
        val prefs = getPreferences(MODE_PRIVATE)
        smoothingEnabled = prefs.getBoolean("smoothing", true)
        binding.smoothing.isChecked = smoothingEnabled
        binding.preview.implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        binding.preview.scaleType = PreviewView.ScaleType.FIT_CENTER
        binding.resolution.adapter = ArrayAdapter(this,android.R.layout.simple_spinner_dropdown_item,listOf("256 × 256","384 × 384"))
        val storedSize = if (prefs.contains("size")) prefs.getInt("size", InputSize.DEFAULT) else null
        binding.resolution.setSelection(InputSize.spinnerIndex(InputSize.resolve(storedSize)))
        binding.resolution.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onNothingSelected(parent: AdapterView<*>?) {}
            override fun onItemSelected(parent: AdapterView<*>?,view: View?,position: Int,id: Long) {
                val size = if (position==0) 256 else 384
                prefs.edit().putInt("size",size).apply()
                generation++
                slotPool = null
                preprocessor = null
                loadPipeline(size,generation)
            }
        }
        binding.smoothing.setOnCheckedChangeListener { _,enabled ->
            smoothingEnabled = enabled
            prefs.edit().putBoolean("smoothing",enabled).apply()
            postExecutor.tryExecute { smoother.reset(); colorizer?.reset(); unsmoothedPreviousRgb=null }
        }
        binding.bokeh.setOnCheckedChangeListener { _,enabled ->
            bokehEnabled = enabled
            binding.depthLabel.text = if (enabled) "Bokeh 景深 · 点击画面对焦" else "相对逆深度 · 红近 蓝远"
            binding.depth.contentDescription = if (enabled) "深度驱动的景深渲染" else "实时相对逆深度图"
            postExecutor.tryExecute { bokehRenderer?.reset(); colorizer?.reset() }
        }
        binding.depth.setOnTouchListener { view,event ->
            if (!bokehEnabled) return@setOnTouchListener false
            if (event.actionMasked == MotionEvent.ACTION_UP) {
                val side = min(view.width,view.height).toFloat()
                if (side > 0f) {
                    val u = ((event.x - (view.width - side) / 2f) / side).coerceIn(0f,1f)
                    val v = ((event.y - (view.height - side) / 2f) / side).coerceIn(0f,1f)
                    focusPoint = FocusPoint(u.toDouble(),v.toDouble(),focusPoint.version + 1)
                }
                view.performClick()
            }
            true
        }
        binding.retry.setOnClickListener {
            if (hasCameraPermission()) {
                val size = if (binding.resolution.selectedItemPosition==0) 256 else 384
                generation++
                slotPool = null
                preprocessor = null
                loadPipeline(size,generation)
                startCamera()
            } else if (permissionAsked && !shouldShowRequestPermissionRationale(Manifest.permission.CAMERA)) {
                startActivity(Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,Uri.parse("package:$packageName")))
            } else requestPermission()
        }
        binding.benchmark.setOnClickListener { startBenchmark() }
        binding.share.setOnClickListener { shareMeasurements() }
        if (hasCameraPermission()) startCamera() else requestPermission()
    }

    private fun hasCameraPermission() = ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED
    private fun requestPermission() { permissionAsked=true; permissionRequest.launch(Manifest.permission.CAMERA) }

    private fun ExecutorService.tryExecute(block: () -> Unit) {
        if (isShutdown) return
        try { execute(block) } catch (_: RejectedExecutionException) {}
    }

    private fun onInference(block: () -> Unit) {
        if (Thread.currentThread() === inferenceThread) block() else inferenceExecutor.tryExecute(block)
    }

    private fun releaseBitmap(bitmap: Bitmap?) {
        if (bitmap == null || bitmap.isRecycled) return
        postExecutor.tryExecute {
            val current = colorizer
            if (current == null) bitmap.recycle() else current.release(bitmap)
        }
    }

    private fun discardPendingDisplay() {
        pendingDisplay.getAndSet(null)?.bitmap?.let { releaseBitmap(it) }
    }

    private fun publishDisplay(frame: PendingDisplay) {
        val previous = pendingDisplay.getAndSet(frame)
        if (previous == null) runOnUiThread { showPendingDisplay() }
        else if (previous.bitmap !== frame.bitmap) releaseBitmap(previous.bitmap)
    }

    private fun showPendingDisplay() {
        val frame = pendingDisplay.getAndSet(null) ?: return
        if (!resumed || frame.token != generation || isDestroyed) {
            releaseBitmap(frame.bitmap)
            return
        }
        val replaced = displayedBitmap
        displayedBitmap = frame.bitmap
        binding.depth.setImageBitmap(frame.bitmap)
        if (replaced != null && replaced !== frame.bitmap) binding.depth.post { binding.depth.post { releaseBitmap(replaced) } }
        val shown = SystemClock.elapsedRealtimeNanos()
        displayedTimes.addLast(shown)
        while (displayedTimes.size > 30) displayedTimes.removeFirst()
        val fps = if (displayedTimes.size > 1) (displayedTimes.size - 1) * 1e9 / (displayedTimes.last - displayedTimes.first) else 0.0
        binding.stats.text = String.format(Locale.CHINA, "深度更新 %.1f FPS · 处理至显示 %.1f ms\n预处理 %.1f · 推理 %.1f · 后处理 %.1f ms\n%s", fps, (shown - frame.startedNs) / 1e6, frame.timing.preprocessMs, frame.timing.inferenceMs, frame.timing.postprocessMs, frame.label)
        binding.modelInfo.text = "${BuildConfig.MODEL_FLAVOR} · ${frame.engine.size} · ${frame.engine.backend} · SHA ${frame.engine.modelSha256.take(10)}"
        postExecutor.tryExecute {
            if (frame.token == generation && frame.recordable && recorder.add(frame.timing, shown, frame.engine, frame.smoothing)) {
                finishBenchmark(true, "completed_300s")
            }
        }
    }

    private fun loadPipeline(size: Int, token: Int) {
        discardPendingDisplay()
        binding.benchmark.isEnabled=false
        binding.modelInfo.text="载入 $size 模型…"
        onInference {
            if (token != generation) return@onInference
            try {
                while (true) { inferHandoff.poll()?.release() ?: break }
                engine?.close(); engine = null
                val created = DepthEngine(this, size)
                if (token != generation) { created.close(); return@onInference }
                engine = created
                val loadedPreprocessor = YuvPreprocessor(size)
                val pool = SlotPool(size, token)
                postExecutor.tryExecute {
                    if (token != generation) return@tryExecute
                    val previous = colorizer
                    colorizer = DepthColorizer(this, size)
                    bokehRenderer = BokehRenderer(size)
                    previous?.clear()
                    smoother.reset()
                    unsmoothedPreviousRgb = null
                }
                if (token != generation) { created.close(); engine = null; return@onInference }
                preprocessor = loadedPreprocessor
                slotPool = pool
                processingFailed = false
                val description = "${BuildConfig.MODEL_FLAVOR} · $size · ${created.backend} · SHA ${created.modelSha256.take(10)}"
                runOnUiThread {
                    if (token == generation) {
                        binding.modelInfo.text = description
                        binding.benchmark.isEnabled = true
                        displayedTimes.clear()
                    }
                }
            } catch (exception: Exception) {
                processingFailed = true
                runOnUiThread { binding.modelInfo.text="模型载入失败：${exception.message}"; binding.stats.text="点击重试。APK 必须包含真实且接口正确的 ONNX 模型。" }
            }
        }
    }

    private fun startCamera() {
        if (!hasCameraPermission() || isDestroyed || cameraBindingPending) return
        cameraBindingPending=true
        val future=ProcessCameraProvider.getInstance(this)
        future.addListener({
            cameraBindingPending=false
            if (isDestroyed || !hasCameraPermission()) return@addListener
            try {
                val provider=future.get(); cameraProvider=provider
                provider.unbindAll()
                val rotation=binding.preview.display?.rotation ?: 0
                val resolution=ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(Size(640,480),ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)).build()
                val preview=Preview.Builder().setTargetRotation(rotation).setResolutionSelector(resolution).build()
                preview.setSurfaceProvider(binding.preview.surfaceProvider)
                val useAnalysis=ImageAnalysis.Builder().setTargetRotation(rotation).setResolutionSelector(resolution)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
                analysis=useAnalysis
                useAnalysis.setAnalyzer(cameraExecutor) { image ->
                    var slot: FrameSlot? = null
                    try {
                        if (!resumed || processingFailed) return@setAnalyzer
                        val token = generation
                        val currentPreprocessor = preprocessor
                        val pool = slotPool
                        if (currentPreprocessor == null || pool == null || pool.epoch != token || currentPreprocessor.size != pool.size) return@setAnalyzer
                        val acquired = pool.acquire() ?: return@setAnalyzer
                        slot = acquired
                        if (token != generation) { acquired.release(); slot = null; return@setAnalyzer }
                        val started = SystemClock.elapsedRealtimeNanos()
                        currentPreprocessor.process(image, acquired.rgb)
                        acquired.token = token
                        acquired.startedNs = started
                        acquired.preprocessMs = (SystemClock.elapsedRealtimeNanos() - started) / 1e6
                        acquired.recordable = false
                        slot = null
                        inferHandoff.submitLatest(acquired)
                        inferenceExecutor.tryExecute { drainInference() }
                    } catch (exception: Exception) {
                        slot?.release()
                        handleFailure(exception)
                    } finally { image.close() }
                }
                val group=UseCaseGroup.Builder().addUseCase(preview).addUseCase(useAnalysis)
                    .setViewPort(ViewPort.Builder(Rational(1,1),rotation).setScaleType(ViewPort.FILL_CENTER).build()).build()
                provider.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,group)
            } catch (exception: Exception) { binding.stats.text="摄像头失败：${exception.message}" }
        },ContextCompat.getMainExecutor(this))
    }

    private fun drainInference() {
        while (true) {
            val slot = inferHandoff.poll() ?: return
            try {
                val current = engine
                if (current == null || slot.token != generation || slot.size != current.size || processingFailed) {
                    slot.release()
                    continue
                }
                val started = SystemClock.elapsedRealtimeNanos()
                current.inferInto(slot.rgb, slot.depth)
                slot.inferenceMs = (SystemClock.elapsedRealtimeNanos() - started) / 1e6
                if (slot.token != generation || processingFailed) { slot.release(); continue }
                advanceWarmup(current)
                slot.recordable = recordFrames && slot.startedNs >= recordingStartNs
                postHandoff.submitLatest(slot)
                postExecutor.tryExecute { drainPost() }
            } catch (exception: Exception) {
                slot.release()
                handleFailure(exception)
                return
            }
        }
    }

    private fun drainPost() {
        val slot = postHandoff.poll() ?: return
        var released = false
        var bitmap: Bitmap? = null
        try {
            val currentEngine = engine
            val currentColorizer = colorizer
            if (currentEngine == null || currentColorizer == null || !currentColorizer.accepts(slot.depth.size) ||
                slot.token != generation || slot.size != currentEngine.size || processingFailed) {
                slot.release()
                return
            }
                val started = SystemClock.elapsedRealtimeNanos()
                val smoothing = smoothingEnabled
                val smoothed = if (smoothing) {
                    unsmoothedPreviousRgb = null
                    smoother.process(slot.rgb, slot.depth, slot.size, slot.size)
                } else unsmoothed(slot.rgb, slot.depth)
                val bokeh = bokehRenderer?.takeIf { bokehEnabled && it.width == slot.size }
                bitmap = if (bokeh != null) {
                    if (bokehPixels.size != slot.depth.size) bokehPixels = IntArray(slot.depth.size)
                    val point = focusPoint
                    val snap = smoothed.sceneReset || point.version != appliedFocusVersion
                    appliedFocusVersion = point.version
                    bokeh.render(slot.rgb, smoothed.depth, point.u, point.v, snap, bokehPixels)
                    currentColorizer.present(bokehPixels)
                } else currentColorizer.render(smoothed.depth, smoothed.sceneReset)
                val token = slot.token
                val startedNs = slot.startedNs
                val recordable = slot.recordable
                val timing = FrameTiming(startedNs, slot.preprocessMs, slot.inferenceMs, (SystemClock.elapsedRealtimeNanos() - started) / 1e6)
                slot.release()
                released = true
                publishDisplay(PendingDisplay(bitmap, token, startedNs, timing, stageLabel(SystemClock.elapsedRealtimeNanos()), currentEngine, smoothing, recordable))
        } catch (exception: Exception) {
            if (!released) slot.release()
            bitmap?.let { releaseBitmap(it) }
            handleFailure(exception)
        }
    }

    private fun unsmoothed(rgb: FloatArray, depth: FloatArray): SmoothedDepth {
        smoother.reset()
        val previous = unsmoothedPreviousRgb
        var delta = 0.0
        if (previous != null && previous.size == rgb.size) {
            var sum = 0.0
            var index = 0
            while (index < rgb.size) {
                sum += abs(rgb[index].toDouble() - previous[index])
                index++
            }
            delta = sum / rgb.size
        }
        val reset = previous == null || previous.size != rgb.size || delta > .25
        if (previous == null || previous.size != rgb.size) unsmoothedPreviousRgb = FloatArray(rgb.size)
        System.arraycopy(rgb, 0, unsmoothedPreviousRgb!!, 0, rgb.size)
        return SmoothedDepth(depth, reset, delta)
    }

    private fun stageLabel(nowNs: Long) = when {
        profileWarmup > 0 -> "算子后端采样预热…"
        measureWarmup > 0 -> "测速预热…"
        recorder.active -> "测试中：剩余 ${recorder.remainingSeconds(nowNs)} 秒"
        else -> measurementHint
    }

    private fun handleFailure(exception: Exception) {
        processingFailed = true
        while (true) { inferHandoff.poll()?.release() ?: break }
        postExecutor.tryExecute {
            while (true) { postHandoff.poll()?.release() ?: break }
            smoother.reset(); colorizer?.reset(); unsmoothedPreviousRgb = null
        }
        runOnUiThread { if (!isDestroyed) binding.stats.text = "处理失败：${exception.message}。点击重试。" }
        if (benchmarkOpen.get() || recorder.active || measurementPending) finishBenchmark(false, "processing_error: ${exception.message}")
    }

    private fun startBenchmark() {
        if (!hasCameraPermission() || analysis==null || processingFailed) { binding.stats.text="请先启用摄像头并确认深度图正常更新。"; return }
        if (bokehEnabled) { binding.stats.text="请先关闭景深模式再测速，测速只统计深度图链路。"; return }
        if (!benchmarkOpen.compareAndSet(false, true)) return
        measurementPending = true
        recordFrames = false
        measurementHint = ""
        binding.resolution.isEnabled=false; binding.smoothing.isEnabled=false; binding.bokeh.isEnabled=false; binding.benchmark.isEnabled=false; binding.retry.isEnabled=false
        binding.share.isEnabled=false
        onInference {
            try {
                val current = engine ?: error("模型尚未载入")
                val folder = recorder.prepare()
                recorder.noteEngine(current, smoothingEnabled)
                current.beginProfiling(File(folder, "ort_provider_sample"))
                profiling = true
                profileWarmup = 10
                measureWarmup = 0
                providerProfile = null
                postExecutor.tryExecute { smoother.reset(); colorizer?.reset(); unsmoothedPreviousRgb = null }
            } catch (exception: Exception) { finishBenchmark(false, "profile_error: ${exception.message}") }
        }
    }

    private fun advanceWarmup(current: DepthEngine) {
        if (profileWarmup > 0) {
            profileWarmup--
            if (profileWarmup == 0) {
                providerProfile = current.endProfiling()
                profiling = false
                measureWarmup = 10
            }
        } else if (measureWarmup > 0) {
            measureWarmup--
            if (measureWarmup == 0) {
                val now = SystemClock.elapsedRealtimeNanos()
                recorder.start(now, current, smoothingEnabled, providerProfile)
                recordingStartNs = now
                recordFrames = true
                measurementPending = false
            }
        }
    }

    private fun finishBenchmark(complete: Boolean, reason: String) {
        if (!benchmarkOpen.compareAndSet(true, false)) return
        onInference {
            if (profiling) {
                try { engine?.endProfiling() } catch (_: Exception) {}
                profiling = false
            }
            profileWarmup = 0
            measureWarmup = 0
            recordFrames = false
            measurementPending = false
            measurementFiles = recorder.finish(complete, reason, engine)
            measurementHint = if (complete) "测试完成，可导出 CSV、JSON 与后端采样。" else "测试停止：$reason"
            runOnUiThread {
                binding.resolution.isEnabled=true; binding.smoothing.isEnabled=true; binding.bokeh.isEnabled=true; binding.benchmark.isEnabled=engine!=null; binding.retry.isEnabled=true
                binding.share.isEnabled=true
                binding.stats.text=if (complete) "5 分钟测试完成，点击导出测量获取 CSV、JSON 和后端采样。" else "测试停止：$reason，已记录部分结果。"
            }
        }
    }

    private fun shareMeasurements() {
        postExecutor.tryExecute {
            val files = if (measurementFiles.isNotEmpty()) measurementFiles else recorder.latestFiles()
            runOnUiThread {
                if (files.isEmpty()) { binding.stats.text="先完成测试，再导出测量。"; return@runOnUiThread }
                val uris = ArrayList(files.map { FileProvider.getUriForFile(this, "$packageName.files", it) })
                val intent = Intent(Intent.ACTION_SEND_MULTIPLE).setType("application/octet-stream")
                    .putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                startActivity(Intent.createChooser(intent, "导出实际测量结果"))
            }
        }
    }

    override fun onStart() {
        super.onStart(); resumed=true
        postExecutor.tryExecute { smoother.reset(); colorizer?.reset(); unsmoothedPreviousRgb=null }
        displayedTimes.clear()
        if (hasCameraPermission() && analysis==null) startCamera()
    }

    override fun onStop() {
        resumed=false
        discardPendingDisplay()
        if (benchmarkOpen.get() || recorder.active || measurementPending) finishBenchmark(false, "activity_backgrounded")
        postExecutor.tryExecute { smoother.reset(); colorizer?.reset(); unsmoothedPreviousRgb=null }
        super.onStop()
    }

    override fun onSaveInstanceState(outState: Bundle) { outState.putBoolean("permissionAsked",permissionAsked); super.onSaveInstanceState(outState) }

    override fun onDestroy() {
        generation++
        slotPool = null
        preprocessor = null
        resumed = false
        discardPendingDisplay()
        displayedBitmap = null
        analysis?.clearAnalyzer(); cameraProvider?.unbindAll()
        onInference { engine?.close(); engine=null }
        cameraExecutor.shutdown()
        inferenceExecutor.shutdown()
        postExecutor.shutdown()
        super.onDestroy()
    }
}
