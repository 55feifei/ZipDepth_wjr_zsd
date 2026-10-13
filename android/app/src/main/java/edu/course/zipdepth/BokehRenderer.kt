package edu.course.zipdepth

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.tanh

/**
 * Depth-aware bokeh, mirroring zipdepth/bokeh.py (BokehMe classical scattering renderer).
 * RGB is planar CHW in [0,1]; checked against tests/fixtures/bokeh.json.
 * Buffers are reused between frames, so an instance must stay on one thread.
 */
class BokehRenderer(
    val width: Int,
    val height: Int = width,
    val downsample: Int = 2,
    val strength: Float = .03f,
    val gamma: Float = 4f,
) {
    init { require(downsample >= 1 && width % downsample == 0 && height % downsample == 0) }

    private val pixels = width * height
    private val smallWidth = width / downsample
    private val smallHeight = height / downsample
    private val smallPixels = smallWidth * smallHeight
    /** Smoothed disparity in [0,1] (1 = nearest) from the last [track] call. */
    val disparity = FloatArray(pixels)
    private val defocus = FloatArray(pixels)
    private val smallColor = FloatArray(3 * smallPixels)
    private val smallDefocus = FloatArray(smallPixels)
    private val acc = FloatArray(4 * smallPixels)
    private val blurred = FloatArray(3 * smallPixels)
    private val output = FloatArray(3 * pixels)
    private val sortScratch = FloatArray(pixels)
    private val windowScratch = FloatArray((2 * FOCUS_WINDOW_RADIUS + 1) * (2 * FOCUS_WINDOW_RADIUS + 1))
    private var distances = FloatArray(0)
    private var distanceReach = -1
    private val gammaLut = FloatArray(GAMMA_STEPS + 1) { (it.toDouble() / GAMMA_STEPS).pow(gamma.toDouble()).toFloat() }
    private val upX = UpsampleAxis(width, smallWidth, downsample)
    private val upY = UpsampleAxis(height, smallHeight, downsample)
    private var low: Double? = null
    private var high: Double? = null
    private var focus: Double? = null

    fun reset() { low = null; high = null; focus = null }

    /** Python FocusTracker.update: EMA-smoothed P2/P98 normalisation and focus disparity. */
    fun track(inverseDepth: FloatArray, u: Double, v: Double, snap: Boolean): Float {
        require(inverseDepth.size == pixels)
        System.arraycopy(inverseDepth, 0, sortScratch, 0, pixels)
        sortScratch.sort()
        val newLow = TemporalSmoother.percentile(sortScratch, .02)
        val newHigh = TemporalSmoother.percentile(sortScratch, .98)
        if (snap || low == null) { low = newLow; high = newHigh }
        else { low = low!! * (1 - RANGE_EMA) + newLow * RANGE_EMA; high = high!! * (1 - RANGE_EMA) + newHigh * RANGE_EMA }
        val lo = low!!
        val span = max(high!! - lo, 1e-6)
        for (i in 0 until pixels) disparity[i] = ((inverseDepth[i] - lo) / span).coerceIn(0.0, 1.0).toFloat()
        val target = focusDisparity(disparity, width, height, u, v, windowScratch)
        focus = if (snap || focus == null) target else focus!! * (1 - FOCUS_EMA) + target * FOCUS_EMA
        return focus!!.toFloat()
    }

    /** Tracks focus at (u, v) in [0,1] and writes opaque ARGB pixels. */
    fun render(rgb: FloatArray, inverseDepth: FloatArray, u: Double, v: Double, snap: Boolean, argb: IntArray) {
        require(argb.size == pixels)
        val color = renderDisparity(rgb, disparity, track(inverseDepth, u, v, snap))
        for (i in 0 until pixels) {
            val r = (color[i] * 255f + .5f).toInt()
            val g = (color[pixels + i] * 255f + .5f).toInt()
            val b = (color[2 * pixels + i] * 255f + .5f).toInt()
            argb[i] = (255 shl 24) or (r shl 16) or (g shl 8) or b
        }
    }

    /** Python render_bokeh. The returned CHW buffer is overwritten by the next call. */
    fun renderDisparity(rgb: FloatArray, disparity: FloatArray, focus: Float): FloatArray {
        require(rgb.size == 3 * pixels && disparity.size == pixels)
        val scale = strength * max(width, height)
        for (i in 0 until pixels) defocus[i] = scale * (disparity[i] - focus)
        val area = (downsample * downsample).toFloat()
        for (sy in 0 until smallHeight) for (sx in 0 until smallWidth) {
            val s = sy * smallWidth + sx
            var radius = 0f
            for (c in 0 until 3) {
                var sum = 0f
                for (oy in 0 until downsample) for (ox in 0 until downsample) {
                    val i = (sy * downsample + oy) * width + sx * downsample + ox
                    sum += linear(rgb[c * pixels + i])
                    if (c == 0) radius += defocus[i]
                }
                smallColor[c * smallPixels + s] = sum / area
            }
            smallDefocus[s] = radius / area / downsample
        }
        scatter()
        val inverse = 1.0 / gamma
        for (s in 0 until smallPixels) {
            val weight = acc[s]
            for (c in 0 until 3) {
                val value = acc[(c + 1) * smallPixels + s] / weight
                blurred[c * smallPixels + s] =
                    if (downsample == 1) value.toDouble().pow(inverse).toFloat().coerceIn(0f, 1f)
                    else value.coerceIn(0f, 1f).toDouble().pow(inverse).toFloat()
            }
        }
        if (downsample == 1) {
            System.arraycopy(blurred, 0, output, 0, output.size)
            return output
        }
        for (y in 0 until height) {
            val y0 = upY.low[y] * smallWidth; val y1 = upY.high[y] * smallWidth; val ty = upY.t[y]
            for (x in 0 until width) {
                val i = y * width + x
                val alpha = (abs(defocus[i]) - .5f).coerceIn(0f, 1f)
                val x0 = upX.low[x]; val x1 = upX.high[x]; val tx = upX.t[x]
                for (c in 0 until 3) {
                    val image = rgb[c * pixels + i].coerceIn(0f, 1f)
                    if (alpha == 0f) { output[c * pixels + i] = image; continue }
                    val base = c * smallPixels
                    val top = blurred[base + y0 + x0] * (1 - tx) + blurred[base + y0 + x1] * tx
                    val bottom = blurred[base + y1 + x0] * (1 - tx) + blurred[base + y1 + x1] * tx
                    val up = top * (1 - ty) + bottom * ty
                    output[c * pixels + i] = (image * (1 - alpha) + up * alpha).coerceIn(0f, 1f)
                }
            }
        }
        return output
    }

    private fun linear(value: Float): Float {
        val position = value.coerceIn(0f, 1f) * GAMMA_STEPS
        val index = min(position.toInt(), GAMMA_STEPS - 1)
        val t = position - index
        return gammaLut[index] + (gammaLut[index + 1] - gammaLut[index]) * t
    }

    /** Each pixel spreads over a square of half-size floor(r) + 1, like BokehMe's CUDA kernel. */
    private fun scatter() {
        acc.fill(0f)
        var maxReach = 0
        for (s in 0 until smallPixels) maxReach = max(maxReach, floor(abs(smallDefocus[s])).toInt() + 1)
        val side = 2 * maxReach + 1
        if (distanceReach != maxReach) {
            distances = FloatArray(side * side) { val dy = it / side - maxReach; val dx = it % side - maxReach; sqrt((dy * dy + dx * dx).toDouble()).toFloat() }
            distanceReach = maxReach
        }
        for (y in 0 until smallHeight) for (x in 0 until smallWidth) {
            val s = y * smallWidth + x
            val radius = abs(smallDefocus[s])
            val reach = floor(radius).toInt() + 1
            val norm = 1f / (radius * radius + WEIGHT_EPS)
            val c0 = smallColor[s]; val c1 = smallColor[smallPixels + s]; val c2 = smallColor[2 * smallPixels + s]
            for (dy in max(-reach, -y)..min(reach, smallHeight - 1 - y)) {
                val row = (y + dy) * smallWidth
                val distanceRow = (dy + maxReach) * side + maxReach
                for (dx in max(-reach, -x)..min(reach, smallWidth - 1 - x)) {
                    val weight = soft(radius - distances[distanceRow + dx]) * norm
                    if (weight == 0f) continue
                    val t = row + x + dx
                    acc[t] += weight
                    acc[smallPixels + t] += weight * c0
                    acc[2 * smallPixels + t] += weight * c1
                    acc[3 * smallPixels + t] += weight * c2
                }
            }
        }
    }

    private class UpsampleAxis(n: Int, small: Int, factor: Int) {
        val low = IntArray(n)
        val high = IntArray(n)
        val t = FloatArray(n)
        init {
            for (i in 0 until n) {
                val source = ((i + .5) / factor - .5).coerceIn(0.0, (small - 1).toDouble())
                low[i] = floor(source).toInt()
                high[i] = min(low[i] + 1, small - 1)
                t[i] = (source - low[i]).toFloat()
            }
        }
    }

    companion object {
        const val SOFT_EDGE = 4.0
        const val WEIGHT_EPS = .2f
        const val FOCUS_WINDOW_RADIUS = 4
        const val RANGE_EMA = .1
        const val FOCUS_EMA = .3
        private const val GAMMA_STEPS = 4096
        private const val SOFT_RANGE = 3
        private const val SOFT_STEPS = 1024
        private val softLut = FloatArray(2 * SOFT_RANGE * SOFT_STEPS + 1) {
            (.5 + .5 * tanh(SOFT_EDGE * (it.toDouble() / SOFT_STEPS - SOFT_RANGE))).toFloat()
        }

        /** 0.5 + 0.5 tanh(4x): the anti-aliased disc edge. */
        private fun soft(x: Float): Float {
            if (x <= -SOFT_RANGE) return 0f
            if (x >= SOFT_RANGE) return 1f
            val position = (x + SOFT_RANGE) * SOFT_STEPS
            val index = min(position.toInt(), softLut.size - 2)
            val t = position - index
            return softLut[index] + (softLut[index + 1] - softLut[index]) * t
        }

        /** Python focus_disparity: median of a window around (u, v) in [0,1], round half to even. */
        fun focusDisparity(disparity: FloatArray, width: Int, height: Int, u: Double, v: Double,
                           scratch: FloatArray = FloatArray((2 * FOCUS_WINDOW_RADIUS + 1) * (2 * FOCUS_WINDOW_RADIUS + 1))): Double {
            val x = Math.rint(u.coerceIn(0.0, 1.0) * (width - 1)).toInt()
            val y = Math.rint(v.coerceIn(0.0, 1.0) * (height - 1)).toInt()
            var count = 0
            for (yy in max(y - FOCUS_WINDOW_RADIUS, 0)..min(y + FOCUS_WINDOW_RADIUS, height - 1))
                for (xx in max(x - FOCUS_WINDOW_RADIUS, 0)..min(x + FOCUS_WINDOW_RADIUS, width - 1))
                    scratch[count++] = disparity[yy * width + xx]
            val window = scratch.copyOf(count)
            window.sort()
            return TemporalSmoother.percentile(window, .5)
        }
    }
}
