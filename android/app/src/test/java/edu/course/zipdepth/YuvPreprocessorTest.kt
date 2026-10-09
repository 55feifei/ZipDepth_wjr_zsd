package edu.course.zipdepth

import org.junit.Assert.assertArrayEquals
import org.junit.Test
import kotlin.math.floor

class YuvPreprocessorTest {
    @Test fun matchesCenterCropBilinearReference() {
        val width = 8
        val height = 8
        val y = ByteArray(width * height) { index -> (16 + index * 3).toByte() }
        val u = ByteArray(width * height / 4) { index -> (96 + index).toByte() }
        val v = ByteArray(width * height / 4) { index -> (160 + index * 2).toByte() }
        val preprocessor = YuvPreprocessor(2)
        for (rotation in intArrayOf(0, 90, 180, 270)) {
            val actual = FloatArray(3 * 2 * 2)
            preprocessor.processPlanes(y, width, 1, u, width / 2, 1, v, width / 2, 1, 1, 1, 5, 6, rotation, actual)
            preprocessor.processPlanes(y, width, 1, u, width / 2, 1, v, width / 2, 1, 1, 1, 5, 6, rotation, actual)
            assertArrayEquals(reference(2, y, width, 1, u, width / 2, 1, v, width / 2, 1, 1, 1, 5, 6, rotation), actual, 0f)
        }
    }

    private fun reference(
        size: Int, y: ByteArray, yRow: Int, yPix: Int, u: ByteArray, uRow: Int, uPix: Int, v: ByteArray, vRow: Int, vPix: Int,
        cropLeft: Int, cropTop: Int, cropWidth: Int, cropHeight: Int, rotation: Int,
    ): FloatArray {
        val yPlane = YuvPlane(y, yRow, yPix)
        val uPlane = YuvPlane(u, uRow, uPix)
        val vPlane = YuvPlane(v, vRow, vPix)
        val side = minOf(cropWidth, cropHeight)
        val left = cropLeft + (cropWidth - side) / 2
        val top = cropTop + (cropHeight - side) / 2
        val area = size * size
        val rgb = FloatArray(3 * area)
        for (oy in 0 until size) for (ox in 0 until size) {
            val fx = ((ox + .5) * side / size - .5).coerceIn(0.0, (side - 1).toDouble())
            val fy = ((oy + .5) * side / size - .5).coerceIn(0.0, (side - 1).toDouble())
            val x0 = floor(fx).toInt(); val y0 = floor(fy).toInt()
            val x1 = (x0 + 1).coerceAtMost(side - 1); val y1 = (y0 + 1).coerceAtMost(side - 1)
            val wx = fx - x0; val wy = fy - y0
            fun pixel(x: Int, yy: Int): Int {
                val index = YuvSampler.sourceIndex(x, yy, side, rotation)
                return YuvSampler.rgbAt(yPlane, uPlane, vPlane, left + index % side, top + index / side)
            }
            val c00 = pixel(x0, y0); val c10 = pixel(x1, y0); val c01 = pixel(x0, y1); val c11 = pixel(x1, y1)
            val w00 = (1 - wx) * (1 - wy); val w10 = wx * (1 - wy); val w01 = (1 - wx) * wy; val w11 = wx * wy
            for (channel in 0..2) {
                val shift = (2 - channel) * 8
                rgb[channel * area + oy * size + ox] = (((c00 shr shift and 255) * w00 + (c10 shr shift and 255) * w10 +
                    (c01 shr shift and 255) * w01 + (c11 shr shift and 255) * w11) / 255.0).toFloat()
            }
        }
        return rgb
    }
}
