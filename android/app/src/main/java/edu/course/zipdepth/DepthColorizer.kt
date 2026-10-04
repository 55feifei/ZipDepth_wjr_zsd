package edu.course.zipdepth

import android.graphics.Bitmap
import android.content.Context
import org.json.JSONArray
import kotlin.math.max

/** Slowly updated percentile display limits avoid per-frame min/max color flicker. */
class DepthColorizer(context: Context, private val size: Int) {
    private var low: Double? = null
    private var high: Double? = null
    private val pixels = IntArray(size * size)
    private val lut = context.assets.open("turbo_lut.json").bufferedReader().use { reader ->
        val values = JSONArray(reader.readText())
        require(values.length() == 256)
        IntArray(256) { index ->
            val rgb = values.getJSONArray(index)
            (255 shl 24) or (rgb.getInt(0) shl 16) or (rgb.getInt(1) shl 8) or rgb.getInt(2)
        }
    }

    fun reset() { low = null; high = null }

    fun render(depth: FloatArray, sceneReset: Boolean): Bitmap {
        val sorted = depth.copyOf().apply { sort() }
        val newLow = TemporalSmoother.percentile(sorted, .02)
        val newHigh = TemporalSmoother.percentile(sorted, .98)
        if (sceneReset || low == null) { low = newLow; high = newHigh }
        else { low = .95 * low!! + .05 * newLow; high = .95 * high!! + .05 * newHigh }
        val span = max(high!! - low!!, 1e-6)
        depth.indices.forEach { i -> pixels[i] = lut[((depth[i] - low!!) / span * 255).toInt().coerceIn(0, 255)] }
        return Bitmap.createBitmap(pixels, size, size, Bitmap.Config.ARGB_8888)
    }

}
