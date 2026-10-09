package edu.course.zipdepth

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max

data class SmoothedDepth(val depth: FloatArray, val sceneReset: Boolean, val rgbDelta: Double)

/** Matches zipdepth.course.runtime.GatedEMA: RGB is planar CHW in [0,1]. */
class TemporalSmoother {
    private var previousRgb: FloatArray? = null
    private var historyDepth = FloatArray(0)
    private var writeDepth = FloatArray(0)
    private var rgbHistory = FloatArray(0)
    private var sortScratch = FloatArray(0)
    private var previousWidth = 0
    private var previousHeight = 0

    fun reset() {
        previousRgb = null
        historyDepth = FloatArray(0)
        writeDepth = FloatArray(0)
        rgbHistory = FloatArray(0)
        previousWidth = 0
        previousHeight = 0
    }

    fun process(rgb: FloatArray, depth: FloatArray, width: Int, height: Int): SmoothedDepth {
        require(width > 0 && height > 0 && depth.size == width * height && rgb.size == 3 * depth.size)
        require(depth.all { it.isFinite() } && rgb.all { it.isFinite() && it in 0f..1f })
        val previousColor = previousRgb
        val previousOutput = if (historyDepth.size == depth.size) historyDepth else null
        val delta = if (previousColor?.size == rgb.size) {
            rgb.indices.sumOf { abs(rgb[it].toDouble() - previousColor[it]) } / rgb.size
        } else 0.0
        val clear = previousOutput == null || previousColor?.size != rgb.size || previousWidth != width || previousHeight != height || delta > 0.25
        if (writeDepth.size != depth.size || writeDepth === historyDepth) writeDepth = FloatArray(depth.size)
        System.arraycopy(depth, 0, writeDepth, 0, depth.size)
        val result = writeDepth
        if (!clear) {
            if (sortScratch.size != depth.size) sortScratch = FloatArray(depth.size)
            System.arraycopy(depth, 0, sortScratch, 0, depth.size)
            sortScratch.sort()
            val scale = max(percentile(sortScratch, .98) - percentile(sortScratch, .02), 1e-6)
            for (y in 0 until height) for (x in 0 until width) {
                val i = y * width + x
                val colorDelta = (abs(rgb[i].toDouble() - previousColor[i]) +
                    abs(rgb[depth.size + i].toDouble() - previousColor[depth.size + i]) +
                    abs(rgb[2 * depth.size + i].toDouble() - previousColor[2 * depth.size + i])) / 3.0
                val history = 0.3 * exp(-colorDelta / 0.08)
                val dx = if (x > 0) abs(depth[i].toDouble() - depth[i - 1]) else 0.0
                val dy = if (y > 0) abs(depth[i].toDouble() - depth[i - width]) else 0.0
                if ((dx + dy) / scale <= 0.1) {
                    result[i] = ((1.0 - history) * depth[i] + history * previousOutput[i]).toFloat()
                }
            }
        }
        if (rgbHistory.size != rgb.size) rgbHistory = FloatArray(rgb.size)
        System.arraycopy(rgb, 0, rgbHistory, 0, rgb.size)
        previousRgb = rgbHistory
        historyDepth = result
        writeDepth = if (previousOutput != null && previousOutput !== result) previousOutput else FloatArray(depth.size)
        previousWidth = width
        previousHeight = height
        return SmoothedDepth(result, clear, delta)
    }

    companion object {
        /** NumPy percentile(method="linear"), input must already be sorted. */
        fun percentile(sorted: FloatArray, fraction: Double): Double {
            require(sorted.isNotEmpty() && fraction in 0.0..1.0)
            val position = (sorted.size - 1) * fraction
            val low = floor(position).toInt()
            val high = (low + 1).coerceAtMost(sorted.lastIndex)
            return sorted[low].toDouble() + (position - low) * (sorted[high].toDouble() - sorted[low])
        }
    }
}
