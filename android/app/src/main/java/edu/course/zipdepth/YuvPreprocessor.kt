package edu.course.zipdepth

import androidx.camera.core.ImageProxy
import kotlin.math.floor

/** Plane addressing deliberately honors both rowStride and pixelStride. */
data class YuvPlane(val bytes: ByteArray, val rowStride: Int, val pixelStride: Int) {
    fun sample(x: Int, y: Int): Int = bytes[y * rowStride + x * pixelStride].toInt() and 255
}

object YuvSampler {
    fun rgbAt(y: YuvPlane, u: YuvPlane, v: YuvPlane, x: Int, yy: Int): Int {
        val luma = (y.sample(x, yy) - 16).coerceAtLeast(0)
        val cb = u.sample(x / 2, yy / 2) - 128
        val cr = v.sample(x / 2, yy / 2) - 128
        val r = ((298 * luma + 409 * cr + 128) shr 8).coerceIn(0, 255)
        val g = ((298 * luma - 100 * cb - 208 * cr + 128) shr 8).coerceIn(0, 255)
        val b = ((298 * luma + 516 * cb + 128) shr 8).coerceIn(0, 255)
        return (r shl 16) or (g shl 8) or b
    }

    /** Maps an upright square pixel to its unrotated crop coordinates. */
    fun sourceIndex(x: Int, y: Int, side: Int, rotation: Int): Int = when (rotation) {
        0 -> y * side + x
        90 -> (side - 1 - x) * side + y
        180 -> (side - 1 - y) * side + side - 1 - x
        270 -> x * side + side - 1 - y
        else -> error("Unsupported camera rotation: $rotation")
    }
}

class YuvPreprocessor(private val size: Int) {
    val rgb = FloatArray(3 * size * size)
    private var cachedPlanes = arrayOf(ByteArray(0), ByteArray(0), ByteArray(0))

    fun process(image: ImageProxy): FloatArray {
        require(image.planes.size == 3) { "Expected CameraX YUV_420_888" }
        val planes = image.planes.mapIndexed { index, plane ->
            val buffer = plane.buffer.duplicate()
            if (cachedPlanes[index].size != buffer.remaining()) cachedPlanes[index] = ByteArray(buffer.remaining())
            buffer.get(cachedPlanes[index])
            YuvPlane(cachedPlanes[index], plane.rowStride, plane.pixelStride)
        }
        val crop = image.cropRect
        val side = minOf(crop.width(), crop.height())
        val left = crop.left + (crop.width() - side) / 2
        val top = crop.top + (crop.height() - side) / 2
        val rotation = image.imageInfo.rotationDegrees
        val area = size * size
        fun pixel(x: Int, yy: Int): Int {
            val index = YuvSampler.sourceIndex(x, yy, side, rotation)
            return YuvSampler.rgbAt(planes[0], planes[1], planes[2], left + index % side, top + index / side)
        }
        for (oy in 0 until size) for (ox in 0 until size) {
            // OpenCV-style half-pixel bilinear sampling, after rotation and square crop.
            val fx = ((ox + .5) * side / size - .5).coerceIn(0.0, (side - 1).toDouble())
            val fy = ((oy + .5) * side / size - .5).coerceIn(0.0, (side - 1).toDouble())
            val x0 = floor(fx).toInt(); val y0 = floor(fy).toInt()
            val x1 = (x0 + 1).coerceAtMost(side - 1); val y1 = (y0 + 1).coerceAtMost(side - 1)
            val wx = fx - x0; val wy = fy - y0
            val c00 = pixel(x0,y0); val c10 = pixel(x1,y0); val c01 = pixel(x0,y1); val c11 = pixel(x1,y1)
            val w00 = (1-wx)*(1-wy); val w10 = wx*(1-wy); val w01 = (1-wx)*wy; val w11 = wx*wy
            for (channel in 0..2) {
                val shift = (2-channel)*8
                rgb[channel * area + oy * size + ox] = (((c00 shr shift and 255)*w00 + (c10 shr shift and 255)*w10 +
                    (c01 shr shift and 255)*w01 + (c11 shr shift and 255)*w11) / 255.0).toFloat()
            }
        }
        return rgb
    }
}
