package edu.course.zipdepth

import com.google.gson.JsonObject
import com.google.gson.JsonParser
import org.junit.Assert.*
import org.junit.Test

class BokehRendererTest {
    private val fixture: JsonObject = javaClass.classLoader!!.getResourceAsStream("bokeh.json")
        ?.bufferedReader()?.use { JsonParser.parseReader(it).asJsonObject }
        ?: error("Shared Python fixture tests/fixtures/bokeh.json is required")
    private val size = fixture.get("width").asInt
    private val strength = fixture.get("strength").asFloat

    private fun JsonObject.floats(key: String) = getAsJsonArray(key).map { it.asFloat }.toFloatArray()
    private fun JsonObject.point() = getAsJsonArray("point").let { it[0].asDouble to it[1].asDouble }

    @Test fun sharedPythonGoldenRenders() {
        val rgb = fixture.floats("rgb")
        val disparity = fixture.floats("disparity")
        for (element in fixture.getAsJsonArray("cases")) {
            val case = element.asJsonObject
            val name = case.get("name").asString
            val (u, v) = case.point()
            val focus = BokehRenderer.focusDisparity(disparity, size, size, u, v)
            assertEquals(name, case.get("focus").asDouble, focus, 1e-6)
            val renderer = BokehRenderer(size, size, case.get("downsample").asInt, strength, case.get("gamma").asFloat)
            assertArrayEquals(name, case.floats("expected"), renderer.renderDisparity(rgb, disparity, focus.toFloat()), 5e-4f)
        }
    }

    @Test fun sharedPythonFocusTracking() {
        val renderer = BokehRenderer(size, size, 2, strength)
        for ((index, element) in fixture.getAsJsonArray("tracker").withIndex()) {
            val frame = element.asJsonObject
            val (u, v) = frame.point()
            val focus = renderer.track(frame.floats("inverse_depth"), u, v, frame.get("snap").asBoolean)
            assertEquals("frame $index", frame.get("focus").asDouble, focus.toDouble(), 1e-5)
            assertArrayEquals("frame $index", frame.floats("disparity"), renderer.disparity, 1e-5f)
        }
    }

    @Test fun inFocusPixelsPassThroughAndOutputIsOpaque() {
        val rgb = FloatArray(3 * 64) { (it % 7) / 7f }
        val inverse = FloatArray(64) { if (it % 8 < 4) 2f else 1f }
        val renderer = BokehRenderer(8, 8, 2, .25f)
        val argb = IntArray(64)
        renderer.render(rgb, inverse, 0.0, 0.0, true, argb)
        for (y in 0 until 8) for (x in 0 until 4) {
            val i = y * 8 + x
            val expected = (255 shl 24) or ((rgb[i] * 255f + .5f).toInt() shl 16) or
                ((rgb[64 + i] * 255f + .5f).toInt() shl 8) or (rgb[128 + i] * 255f + .5f).toInt()
            assertEquals("pixel $i", expected, argb[i])
        }
        assertTrue(argb.all { it ushr 24 == 255 })
    }

    @Test fun rejectsSizesThatTheDownsampleDoesNotDivide() {
        try { BokehRenderer(10, 10, 4); fail("10 is not divisible by 4") } catch (_: IllegalArgumentException) {}
    }
}
