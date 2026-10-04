package edu.course.zipdepth

import com.google.gson.JsonParser
import org.junit.Assert.*
import org.junit.Test

class TemporalSmootherTest {
    @Test fun sharedPythonGoldenSequences() {
        val stream=javaClass.classLoader!!.getResourceAsStream("smoothing.json") ?: error("Shared Python fixture is required")
        val fixture=stream.bufferedReader().use { JsonParser.parseReader(it).asJsonObject }
        for (case in fixture.getAsJsonArray("cases")) {
            val value=case.asJsonObject
            val width=value.get("width").asInt; val height=value.get("height").asInt
            val smoother=TemporalSmoother()
            for (frame in value.getAsJsonArray("frames")) {
                val sample=frame.asJsonObject
                fun floats(key:String)=sample.getAsJsonArray(key).map { it.asFloat }.toFloatArray()
                val result=smoother.process(floats("rgb"),floats("depth"),width,height)
                assertArrayEquals(value.get("name").asString,floats("expected"),result.depth,1e-6f)
                assertEquals(value.get("name").asString,sample.get("scene_reset").asBoolean,result.sceneReset)
            }
        }
    }

    @Test fun rejectsNonFiniteDepthWithoutPoisoningHistory() {
        val smoother=TemporalSmoother(); val rgb=FloatArray(12){.5f}
        smoother.process(rgb,floatArrayOf(1f,1f,1f,1f),2,2)
        try { smoother.process(rgb,floatArrayOf(Float.NaN,1f,1f,1f),2,2); fail("NaN accepted") } catch (_:IllegalArgumentException) {}
        assertArrayEquals(floatArrayOf(1.7f,1.7f,1.7f,1.7f),smoother.process(rgb,FloatArray(4){2f},2,2).depth,1e-6f)
    }

    @Test fun resetDiscardsHistory() {
        val smoother=TemporalSmoother(); val rgb=FloatArray(12){.5f}
        smoother.process(rgb,FloatArray(4){1f},2,2); smoother.reset()
        val result=smoother.process(rgb,FloatArray(4){2f},2,2)
        assertTrue(result.sceneReset); assertArrayEquals(FloatArray(4){2f},result.depth,0f)
    }
}
