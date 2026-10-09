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
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private val worker = Executors.newSingleThreadExecutor()
    private var engine: DepthEngine? = null
    private var preprocessor: YuvPreprocessor? = null
    private var colorizer: DepthColorizer? = null
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
    private var profileWarmup = 0
    private var measureWarmup = 0
    private var providerProfile: File? = null
    private var measurementFiles: List<File> = emptyList()
    private val displayedTimes = ArrayDeque<Long>()
    private val pendingDisplay = AtomicReference<PendingDisplay?>(null)
    private var permissionAsked = false

    private data class PendingDisplay(
        val bitmap: Bitmap,
        val token: Int,
        val startedNs: Long,
        val timing: FrameTiming,
        val label: String,
        val engine: DepthEngine,
        val smoothing: Boolean,
    )

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
                loadPipeline(size,generation)
            }
        }
        binding.smoothing.setOnCheckedChangeListener { _,enabled ->
            smoothingEnabled = enabled
            prefs.edit().putBoolean("smoothing",enabled).apply()
            worker.execute { smoother.reset(); colorizer?.reset(); unsmoothedPreviousRgb=null }
        }
        binding.retry.setOnClickListener {
            if (hasCameraPermission()) {
                val size = if (binding.resolution.selectedItemPosition==0) 256 else 384
                generation++; loadPipeline(size,generation); startCamera()
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

    private fun discardPendingDisplay() {
        pendingDisplay.getAndSet(null)?.bitmap?.let { bitmap -> if (!bitmap.isRecycled) bitmap.recycle() }
    }

    private fun publishDisplay(frame: PendingDisplay) {
        val previous = pendingDisplay.getAndSet(frame)
        if (previous == null) runOnUiThread { showPendingDisplay() }
        else if (previous.bitmap !== frame.bitmap && !previous.bitmap.isRecycled) previous.bitmap.recycle()
    }

    private fun showPendingDisplay() {
        val frame = pendingDisplay.getAndSet(null) ?: return
        if (!resumed || frame.token != generation || isDestroyed) {
            if (!frame.bitmap.isRecycled) frame.bitmap.recycle()
            return
        }
        binding.depth.setImageBitmap(frame.bitmap)
        val shown = SystemClock.elapsedRealtimeNanos()
        displayedTimes.addLast(shown)
        while (displayedTimes.size > 30) displayedTimes.removeFirst()
        val fps = if (displayedTimes.size > 1) (displayedTimes.size - 1) * 1e9 / (displayedTimes.last - displayedTimes.first) else 0.0
        binding.stats.text = String.format(Locale.CHINA, "深度更新 %.1f FPS · 处理至显示 %.1f ms\n预处理 %.1f · 推理 %.1f · 后处理 %.1f ms\n%s", fps, (shown - frame.startedNs) / 1e6, frame.timing.preprocessMs, frame.timing.inferenceMs, frame.timing.postprocessMs, frame.label)
        binding.modelInfo.text = "${BuildConfig.MODEL_FLAVOR} · ${frame.engine.size} · ${frame.engine.backend} · SHA ${frame.engine.modelSha256.take(10)}"
        if (!worker.isShutdown) worker.execute {
            if (frame.token == generation && recorder.add(frame.timing, shown, frame.engine, frame.smoothing)) finishBenchmark(true, "completed_300s")
        }
    }

    private fun loadPipeline(size: Int,token: Int) {
        discardPendingDisplay()
        binding.benchmark.isEnabled=false
        binding.modelInfo.text="载入 $size 模型…"
        worker.execute {
            if (token!=generation) return@execute
            try {
                engine?.close(); engine=null
                engine=DepthEngine(this,size)
                preprocessor=YuvPreprocessor(size)
                colorizer=DepthColorizer(this,size)
                smoother.reset()
                unsmoothedPreviousRgb=null
                processingFailed=false
                val description="${BuildConfig.MODEL_FLAVOR} · $size · ${engine!!.backend} · SHA ${engine!!.modelSha256.take(10)}"
                runOnUiThread {
                    if (token==generation) {
                        binding.modelInfo.text=description
                        binding.benchmark.isEnabled=true
                        displayedTimes.clear()
                    }
                }
            } catch (exception: Exception) {
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
                useAnalysis.setAnalyzer(worker) { image ->
                    val token=generation
                    try {
                        val currentEngine=engine ?: return@setAnalyzer
                        if (!resumed || processingFailed) return@setAnalyzer
                        val started=SystemClock.elapsedRealtimeNanos()
                        val rgb=preprocessor!!.process(image)
                        val preEnd=SystemClock.elapsedRealtimeNanos()
                        val raw=currentEngine.infer(rgb)
                        val inferEnd=SystemClock.elapsedRealtimeNanos()
                        val smoothed=if (smoothingEnabled) {
                            unsmoothedPreviousRgb=null
                            smoother.process(rgb,raw,currentEngine.size,currentEngine.size)
                        } else {
                            smoother.reset()
                            val previous=unsmoothedPreviousRgb
                            val delta=if (previous?.size==rgb.size) rgb.indices.sumOf { abs(rgb[it].toDouble()-previous[it]) } / rgb.size else 0.0
                            unsmoothedPreviousRgb=rgb.copyOf()
                            SmoothedDepth(raw,previous?.size!=rgb.size || delta>.25,delta)
                        }
                        val bitmap=colorizer!!.render(smoothed.depth,smoothed.sceneReset)
                        val postEnd=SystemClock.elapsedRealtimeNanos()
                        val timing=FrameTiming(started,(preEnd-started)/1e6,(inferEnd-preEnd)/1e6,(postEnd-inferEnd)/1e6)
                        advanceWarmup(currentEngine)
                        val label=when {
                            profileWarmup>0 -> "算子后端采样预热…"
                            measureWarmup>0 -> "测速预热…"
                            recorder.active -> "测试中：剩余 ${recorder.remainingSeconds(postEnd)} 秒"
                            else -> measurementHint
                        }
                        publishDisplay(PendingDisplay(bitmap,token,started,timing,label,currentEngine,smoothingEnabled))
                    } catch (exception: Exception) {
                        processingFailed=true
                        smoother.reset(); colorizer?.reset(); unsmoothedPreviousRgb=null
                        if (recorder.active || measurementPending) finishBenchmark(false,"inference_error: ${exception.message}")
                        runOnUiThread { binding.stats.text="处理失败：${exception.message}。点击重试。" }
                    } finally { image.close() }
                }
                val group=UseCaseGroup.Builder().addUseCase(preview).addUseCase(useAnalysis)
                    .setViewPort(ViewPort.Builder(Rational(1,1),rotation).setScaleType(ViewPort.FILL_CENTER).build()).build()
                provider.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,group)
            } catch (exception: Exception) { binding.stats.text="摄像头失败：${exception.message}" }
        },ContextCompat.getMainExecutor(this))
    }

    private fun startBenchmark() {
        if (measurementPending || recorder.active) return
        if (!hasCameraPermission() || analysis==null || processingFailed) { binding.stats.text="请先启用摄像头并确认深度图正常更新。"; return }
        measurementPending=true
        measurementHint=""
        binding.resolution.isEnabled=false; binding.smoothing.isEnabled=false; binding.benchmark.isEnabled=false; binding.retry.isEnabled=false
        binding.share.isEnabled=false
        worker.execute {
            try {
                val current=engine ?: error("模型尚未载入")
                val folder=recorder.prepare()
                recorder.noteEngine(current,smoothingEnabled)
                current.beginProfiling(File(folder,"ort_provider_sample"))
                profileWarmup=10; measureWarmup=0; providerProfile=null
                smoother.reset(); colorizer?.reset(); unsmoothedPreviousRgb=null
            } catch (exception: Exception) { finishBenchmark(false,"profile_error: ${exception.message}") }
        }
    }

    private fun advanceWarmup(current: DepthEngine) {
        if (profileWarmup>0) {
            profileWarmup--
            if (profileWarmup==0) { providerProfile=current.endProfiling(); measureWarmup=10 }
        } else if (measureWarmup>0) {
            measureWarmup--
            if (measureWarmup==0) {
                recorder.start(SystemClock.elapsedRealtimeNanos(),current,smoothingEnabled,providerProfile)
                measurementPending=false
            }
        }
    }

    private fun finishBenchmark(complete: Boolean,reason: String) {
        val currentEngine=engine
        if (profileWarmup>0) currentEngine?.endProfiling()
        profileWarmup=0; measureWarmup=0; measurementPending=false
        measurementFiles=recorder.finish(complete,reason,currentEngine)
        measurementHint=if (complete) "测试完成，可导出 CSV、JSON 与后端采样。" else "测试停止：$reason"
        runOnUiThread {
            binding.resolution.isEnabled=true; binding.smoothing.isEnabled=true; binding.benchmark.isEnabled=engine!=null; binding.retry.isEnabled=true
            binding.share.isEnabled=true
            binding.stats.text=if (complete) "5 分钟测试完成，点击导出测量获取 CSV、JSON 和后端采样。" else "测试停止：$reason，已记录部分结果。"
        }
    }

    private fun shareMeasurements() {
        worker.execute {
            val files=if (measurementFiles.isNotEmpty()) measurementFiles else recorder.latestFiles()
            runOnUiThread {
                if (files.isEmpty()) { binding.stats.text="先完成测试，再导出测量。"; return@runOnUiThread }
                val uris=ArrayList(files.map { FileProvider.getUriForFile(this,"$packageName.files",it) })
                val intent=Intent(Intent.ACTION_SEND_MULTIPLE).setType("application/octet-stream")
                    .putParcelableArrayListExtra(Intent.EXTRA_STREAM,uris).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                startActivity(Intent.createChooser(intent,"导出实际测量结果"))
            }
        }
    }

    override fun onStart() {
        super.onStart(); resumed=true
        worker.execute { smoother.reset(); colorizer?.reset(); unsmoothedPreviousRgb=null }
        displayedTimes.clear()
        if (hasCameraPermission() && analysis==null) startCamera()
    }

    override fun onStop() {
        resumed=false
        discardPendingDisplay()
        worker.execute {
            if (recorder.active || measurementPending) finishBenchmark(false,"activity_backgrounded")
            smoother.reset(); colorizer?.reset(); unsmoothedPreviousRgb=null
        }
        super.onStop()
    }

    override fun onSaveInstanceState(outState: Bundle) { outState.putBoolean("permissionAsked",permissionAsked); super.onSaveInstanceState(outState) }

    override fun onDestroy() {
        discardPendingDisplay()
        analysis?.clearAnalyzer(); cameraProvider?.unbindAll()
        worker.execute { engine?.close(); engine=null }
        worker.shutdown()
        super.onDestroy()
    }
}
