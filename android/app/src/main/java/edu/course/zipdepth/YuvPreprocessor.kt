package edu.course.zipdepth

import androidx.camera.core.ImageProxy
import kotlin.math.floor

/** Plane addressing deliberately honors both rowStride and pixelStride. */
data class YuvPlane(val bytes: ByteArray, val rowStride: Int, val pixelStride: Int) {
    fun sample(x: Int, y: Int): Int = bytes[y * rowStride + x * pixelStride].toInt() and 255
}

object YuvSampler {
    fun rgb(luma: Int, cb: Int, cr: Int): Int {
        val y = (luma - 16).coerceAtLeast(0)
        val u = cb - 128
        val v = cr - 128
        val r = ((298 * y + 409 * v + 128) shr 8).coerceIn(0, 255)
        val g = ((298 * y - 100 * u - 208 * v + 128) shr 8).coerceIn(0, 255)
        val b = ((298 * y + 516 * u + 128) shr 8).coerceIn(0, 255)
        return (r shl 16) or (g shl 8) or b
    }

    fun rgbAt(y: YuvPlane, u: YuvPlane, v: YuvPlane, x: Int, yy: Int): Int =
        rgb(y.sample(x, yy), u.sample(x / 2, yy / 2), v.sample(x / 2, yy / 2))

    /** Maps an upright square pixel to its unrotated crop coordinates. */
    fun sourceIndex(x: Int, y: Int, side: Int, rotation: Int): Int = when (rotation) {
        0 -> y * side + x
        90 -> (side - 1 - x) * side + y
        180 -> (side - 1 - y) * side + side - 1 - x
        270 -> x * side + side - 1 - y
        else -> error("Unsupported camera rotation: $rotation")
    }
}

class YuvPreprocessor(val size: Int) {
    private var planeCache = arrayOf(ByteArray(0), ByteArray(0), ByteArray(0))
    private var tableSide = -1
    private var axisX0 = IntArray(0)
    private var axisX1 = IntArray(0)
    private var axisY0 = IntArray(0)
    private var axisY1 = IntArray(0)
    private var axisWx = DoubleArray(0)
    private var axisWy = DoubleArray(0)

    fun process(image: ImageProxy, destination: FloatArray): FloatArray {
        require(image.planes.size == 3) { "Expected CameraX YUV_420_888" }
        val planes = image.planes
        val cached = planeCache
        repeat(3) { index ->
            val buffer = planes[index].buffer.duplicate()
            if (cached[index].size != buffer.remaining()) cached[index] = ByteArray(buffer.remaining())
            buffer.get(cached[index])
        }
        val crop = image.cropRect
        return processPlanes(
            cached[0], planes[0].rowStride, planes[0].pixelStride,
            cached[1], planes[1].rowStride, planes[1].pixelStride,
            cached[2], planes[2].rowStride, planes[2].pixelStride,
            crop.left, crop.top, crop.width(), crop.height(), image.imageInfo.rotationDegrees, destination,
        )
    }

    /** Center-square crop, rotation, then the same half-pixel bilinear resize as the previous per-frame loop. */
    fun processPlanes(
        y: ByteArray, yRow: Int, yPix: Int,
        u: ByteArray, uRow: Int, uPix: Int,
        v: ByteArray, vRow: Int, vPix: Int,
        cropLeft: Int, cropTop: Int, cropWidth: Int, cropHeight: Int,
        rotation: Int, destination: FloatArray,
    ): FloatArray {
        val area = size * size
        require(destination.size == 3 * area)
        require(rotation == 0 || rotation == 90 || rotation == 180 || rotation == 270) { "Unsupported camera rotation: $rotation" }
        val side = minOf(cropWidth, cropHeight)
        require(side > 0)
        val left = cropLeft + (cropWidth - side) / 2
        val top = cropTop + (cropHeight - side) / 2
        ensureAxis(side)
        val x0 = axisX0; val x1 = axisX1; val y0 = axisY0; val y1 = axisY1; val wx = axisWx; val wy = axisWy
        var index = 0
        var oy = 0
        while (oy < size) {
            val iy0 = y0[oy]; val iy1 = y1[oy]; val wyy = wy[oy]
            var ox = 0
            while (ox < size) {
                val ix0 = x0[ox]; val ix1 = x1[ox]; val wxx = wx[ox]
                val c00 = sample(y, yRow, yPix, u, uRow, uPix, v, vRow, vPix, ix0, iy0, side, rotation, left, top)
                val c10 = sample(y, yRow, yPix, u, uRow, uPix, v, vRow, vPix, ix1, iy0, side, rotation, left, top)
                val c01 = sample(y, yRow, yPix, u, uRow, uPix, v, vRow, vPix, ix0, iy1, side, rotation, left, top)
                val c11 = sample(y, yRow, yPix, u, uRow, uPix, v, vRow, vPix, ix1, iy1, side, rotation, left, top)
                val w00 = (1.0 - wxx) * (1.0 - wyy)
                val w10 = wxx * (1.0 - wyy)
                val w01 = (1.0 - wxx) * wyy
                val w11 = wxx * wyy
                var channel = 0
                while (channel < 3) {
                    val shift = (2 - channel) shl 3
                    destination[channel * area + index] = (((c00 ushr shift and 255) * w00 + (c10 ushr shift and 255) * w10 +
                        (c01 ushr shift and 255) * w01 + (c11 ushr shift and 255) * w11) / 255.0).toFloat()
                    channel++
                }
                index++
                ox++
            }
            oy++
        }
        return destination
    }

    private fun ensureAxis(side: Int) {
        if (tableSide == side && axisX0.size == size) return
        val x0 = IntArray(size); val x1 = IntArray(size); val y0 = IntArray(size); val y1 = IntArray(size)
        val wx = DoubleArray(size); val wy = DoubleArray(size)
        val last = (side - 1).toDouble()
        for (offset in 0 until size) {
            // Square resize: upright x and y share this half-pixel mapping.
            val coord = ((offset + .5) * side / size - .5).coerceIn(0.0, last)
            val index = floor(coord).toInt()
            x0[offset] = index
            y0[offset] = index
            val next = (index + 1).coerceAtMost(side - 1)
            x1[offset] = next
            y1[offset] = next
            val weight = coord - index
            wx[offset] = weight
            wy[offset] = weight
        }
        axisX0 = x0; axisX1 = x1; axisY0 = y0; axisY1 = y1; axisWx = wx; axisWy = wy
        tableSide = side
    }

    private fun sample(
        y: ByteArray, yRow: Int, yPix: Int,
        u: ByteArray, uRow: Int, uPix: Int,
        v: ByteArray, vRow: Int, vPix: Int,
        uprightX: Int, uprightY: Int, side: Int, rotation: Int, left: Int, top: Int,
    ): Int {
        val x = imageX(uprightX, uprightY, side, rotation, left)
        val yy = imageY(uprightX, uprightY, side, rotation, top)
        val luma = y[yy * yRow + x * yPix].toInt() and 255
        val cb = u[(yy shr 1) * uRow + (x shr 1) * uPix].toInt() and 255
        val cr = v[(yy shr 1) * vRow + (x shr 1) * vPix].toInt() and 255
        return YuvSampler.rgb(luma, cb, cr)
    }

    private fun imageX(x: Int, y: Int, side: Int, rotation: Int, left: Int) = when (rotation) {
        0 -> left + x
        90 -> left + y
        180 -> left + side - 1 - x
        else -> left + side - 1 - y
    }

    private fun imageY(x: Int, y: Int, side: Int, rotation: Int, top: Int) = when (rotation) {
        0 -> top + y
        90 -> top + side - 1 - x
        180 -> top + side - 1 - y
        else -> top + x
    }
}
