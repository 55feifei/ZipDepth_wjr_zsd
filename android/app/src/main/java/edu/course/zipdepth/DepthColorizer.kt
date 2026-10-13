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
    private var sortScratch = FloatArray(0)
    private val spare = ArrayDeque<Bitmap>()
    private val lut = context.assets.open("turbo_lut.json").bufferedReader().use { reader ->
        val values = JSONArray(reader.readText())
        require(values.length() == 256)
        IntArray(256) { index ->
            val rgb = values.getJSONArray(index)
            (255 shl 24) or (rgb.getInt(0) shl 16) or (rgb.getInt(1) shl 8) or rgb.getInt(2)
        }
    }

    fun reset() { low = null; high = null }

    fun accepts(pixelCount: Int) = pixelCount == size * size

    fun clear() {
        reset()
        while (spare.isNotEmpty()) spare.removeFirst().let { if (!it.isRecycled) it.recycle() }
    }

    /** Returned bitmaps stay checked out until [release]. The pool is only touched on the post thread. */
    fun render(depth: FloatArray, sceneReset: Boolean): Bitmap {
        if (sortScratch.size != depth.size) sortScratch = FloatArray(depth.size)
        System.arraycopy(depth, 0, sortScratch, 0, depth.size)
        sortScratch.sort()
        val newLow = TemporalSmoother.percentile(sortScratch, .02)
        val newHigh = TemporalSmoother.percentile(sortScratch, .98)
        if (sceneReset || low == null) { low = newLow; high = newHigh }
        else { low = .95 * low!! + .05 * newLow; high = .95 * high!! + .05 * newHigh }
        val span = max(high!! - low!!, 1e-6)
        val limit = low!!
        var i = 0
        while (i < depth.size) {
            pixels[i] = lut[((depth[i] - limit) / span * 255).toInt().coerceIn(0, 255)]
            i++
        }
        return present(pixels)
    }

    /** Wraps ARGB pixels (for example a bokeh frame) in a pooled bitmap; same checkout rules as [render]. */
    fun present(argb: IntArray): Bitmap {
        require(argb.size == size * size)
        val bitmap = spare.removeFirstOrNull() ?: Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(argb, 0, size, 0, 0, size, size)
        return bitmap
    }

    fun release(bitmap: Bitmap) {
        if (bitmap.isRecycled || spare.contains(bitmap)) return
        if (bitmap.width != size || bitmap.height != size || spare.size >= 4) bitmap.recycle()
        else spare.addFirst(bitmap)
    }

}
