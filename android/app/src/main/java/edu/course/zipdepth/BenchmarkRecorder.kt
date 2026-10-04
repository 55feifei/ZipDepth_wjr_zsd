package edu.course.zipdepth

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.Build
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedWriter
import java.io.File
import java.util.Locale

data class FrameTiming(val analyzedNs: Long, val preprocessMs: Double, val inferenceMs: Double, val postprocessMs: Double)

/** One worker owns this recorder; timestamps for displayed updates come from the UI. */
class BenchmarkRecorder(private val context: Context) {
    private var writer: BufferedWriter? = null
    private var folder: File? = null
    private var metadata = JSONObject()
    private val latencies = ArrayList<Double>()
    private val inference = ArrayList<Double>()
    private var startedNs = 0L
    private var lastDisplayedNs = 0L
    private var firstDisplayedNs = 0L
    var active = false
        private set
    val durationNs = 300_000_000_000L

    fun prepare(): File {
        val destination = File(context.filesDir, "measurements/${System.currentTimeMillis()}").apply { mkdirs() }
        folder = destination
        return destination
    }

    fun start(nowNs: Long, engine: DepthEngine, smoothing: Boolean, profile: File?) {
        check(!active)
        val directory = folder ?: prepare()
        writer = File(directory, "frames.csv").bufferedWriter().also {
            it.write("frame,elapsed_s,preprocess_ms,inference_ms,postprocess_ms,ui_queue_ms,pipeline_ms,display_update_fps,backend,resolution,smoothing,battery_temp_c\n")
        }
        latencies.clear(); inference.clear(); firstDisplayedNs = 0; lastDisplayedNs = 0
        startedNs = nowNs
        metadata = JSONObject().put("manufacturer", Build.MANUFACTURER).put("model", Build.MODEL)
            .put("android_release", Build.VERSION.RELEASE).put("api", Build.VERSION.SDK_INT).put("hardware", Build.HARDWARE)
            .put("model_flavor", BuildConfig.MODEL_FLAVOR).put("model_sha256", engine.modelSha256).put("resolution", engine.size)
            .put("smoothing", smoothing).put("requested_duration_s", 300).put("backend", engine.backend)
            .put("fallback_reason", engine.fallbackReason ?: JSONObject.NULL)
            .put("latency_definition", "ImageProxy analyzer entry to ImageView.setImageBitmap; excludes sensor exposure and screen scanout")
            .put("battery_temp_start_c", batteryTemperature())
        if (Build.VERSION.SDK_INT >= 31) metadata.put("soc", Build.SOC_MODEL)
        if (profile?.isFile == true) {
            metadata.put("profile", profile.name)
            metadata.put("provider_node_events", providerCounts(profile))
        }
        active = true
    }

    fun add(timing: FrameTiming, shownNs: Long, engine: DepthEngine, smoothing: Boolean): Boolean {
        if (!active) return false
        if (shownNs - startedNs > durationNs) return true
        if (firstDisplayedNs == 0L) firstDisplayedNs = shownNs
        lastDisplayedNs = shownNs
        val pipeline = (shownNs - timing.analyzedNs) / 1e6
        val uiQueue = (pipeline - timing.preprocessMs - timing.inferenceMs - timing.postprocessMs).coerceAtLeast(0.0)
        latencies.add(pipeline); inference.add(timing.inferenceMs)
        val fps = if (shownNs > firstDisplayedNs) (latencies.size-1) * 1e9 / (shownNs-firstDisplayedNs) else 0.0
        writer!!.write(String.format(Locale.US, "%d,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s,%d,%s,%.1f\n",
            latencies.size, (shownNs-startedNs)/1e9, timing.preprocessMs, timing.inferenceMs, timing.postprocessMs,
            uiQueue,pipeline,fps,engine.backend,engine.size,smoothing,batteryTemperature()))
        if (latencies.size % 60 == 0) writer!!.flush()
        return false
    }

    fun remainingSeconds(nowNs: Long): Int = ((durationNs-(nowNs-startedNs))/1e9).toInt().coerceAtLeast(0)

    fun finish(complete: Boolean, reason: String): List<File> {
        if (!active) return folder?.listFiles()?.filter { it.isFile } ?: emptyList()
        active = false
        writer?.close(); writer = null
        val elapsed = if (lastDisplayedNs > firstDisplayedNs) (lastDisplayedNs-firstDisplayedNs)/1e9 else 0.0
        metadata.put("completed", complete).put("stop_reason", reason).put("depth_updates",latencies.size)
            .put("observed_update_span_s", elapsed).put("display_update_fps", if (elapsed>0) (latencies.size-1)/elapsed else 0.0)
            .put("pipeline_p50_ms", percentile(latencies,.50)).put("pipeline_p95_ms",percentile(latencies,.95))
            .put("inference_p50_ms",percentile(inference,.50)).put("inference_p95_ms",percentile(inference,.95))
            .put("battery_temp_end_c",batteryTemperature())
        File(folder!!,"summary.json").writeText(metadata.toString(2))
        return folder!!.listFiles()?.filter { it.isFile } ?: emptyList()
    }

    fun latestFiles(): List<File> = File(context.filesDir,"measurements").listFiles()?.filter { it.isDirectory }
        ?.sortedByDescending { it.name }?.firstOrNull()?.listFiles()?.filter { it.isFile } ?: emptyList()

    private fun batteryTemperature(): Double = (context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        ?.getIntExtra("temperature",-1) ?: -1) / 10.0

    private fun percentile(values: List<Double>, q: Double): Any {
        if (values.isEmpty()) return JSONObject.NULL
        val sorted = values.sorted(); val index = (sorted.size-1)*q; val low=index.toInt()
        return sorted[low]+(index-low)*(sorted[(low+1).coerceAtMost(sorted.lastIndex)]-sorted[low])
    }

    private fun providerCounts(file: File): JSONObject {
        val counts = JSONObject()
        try {
            val events=JSONArray(file.readText())
            for (i in 0 until events.length()) {
                val event=events.optJSONObject(i) ?: continue
                val provider=event.optJSONObject("args")?.optString("provider") ?: continue
                if (provider.isNotEmpty()) counts.put(provider,counts.optInt(provider)+1)
            }
        } catch (exception: Exception) { counts.put("parse_error",exception.message) }
        return counts
    }
}
