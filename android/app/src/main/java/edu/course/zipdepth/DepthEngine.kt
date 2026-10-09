package edu.course.zipdepth

import android.content.Context
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

/** All access (creation, inference, profiling, close) stays on the inference thread. */
class DepthEngine(private val context: Context, val size: Int, private val threads: Int = 4) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val model = context.assets.open("zipdepth_${size}.onnx").use { it.readBytes() }
    val modelSha256 = MessageDigest.getInstance("SHA-256").digest(model).joinToString("") { "%02x".format(it) }
    private val input = ByteBuffer.allocateDirect(3 * size * size * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val tensor = OnnxTensor.createTensor(environment, input, longArrayOf(1, 3, size.toLong(), size.toLong()))
    private val feeds = hashMapOf<String, OnnxTensor>("image" to tensor)
    private var session: OrtSession? = null
    private var optionsForSession: OrtSession.SessionOptions? = null
    private var closed = false
    var backend: String = ""
        private set
    var fallbackReason: String? = null
        private set
    private var profilePrefix: File? = null

    init {
        try { session = createWithFallback(null) }
        catch (exception: Exception) { tensor.close(); throw exception }
    }

    private fun createSession(useXnnpack: Boolean, profile: File?): OrtSession {
        val options = OrtSession.SessionOptions()
        try {
            options.setIntraOpNumThreads(if (useXnnpack) 1 else threads)
            options.setInterOpNumThreads(1)
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            options.addConfigEntry("session.intra_op.allow_spinning", "0")
            if (useXnnpack) options.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
            profile?.let { options.enableProfiling(it.absolutePath) }
            val created = environment.createSession(model, options)
            try {
                val inputInfo = created.inputInfo.getValue("image").info as TensorInfo
                val outputInfo = created.outputInfo.getValue("depth").info as TensorInfo
                require(inputInfo.type == OnnxJavaType.FLOAT && inputInfo.shape.contentEquals(longArrayOf(1,3,size.toLong(),size.toLong())))
                require(outputInfo.type == OnnxJavaType.FLOAT && outputInfo.shape.contentEquals(longArrayOf(1,1,size.toLong(),size.toLong())))
                backend = if (useXnnpack) "XNNPACK + CPU" else "CPU"
                optionsForSession = options
                return created
            } catch (exception: Exception) { created.close(); throw exception }
        } catch (exception: Exception) { options.close(); throw exception }
    }

    private fun createWithFallback(profile: File?): OrtSession = try { createSession(true, profile) }
    catch (exception: OrtException) { fallbackReason = exception.message; createSession(false, profile) }

    fun inferInto(rgb: FloatArray, destination: FloatArray) {
        require(rgb.size == input.capacity() && destination.size == size * size)
        input.clear(); input.put(rgb); input.rewind()
        try { runInto(destination) } catch (exception: OrtException) {
            if (backend == "CPU") throw exception
            fallbackReason = exception.message
            closeSession(); session = createSession(false, profilePrefix)
            runInto(destination)
        }
    }

    private fun runInto(destination: FloatArray) {
        (session ?: error("Inference session unavailable")).run(feeds).use { result ->
            val output = result.get("depth").get() as OnnxTensor
            val decoded = output.floatBuffer
            decoded.rewind()
            decoded.get(destination)
        }
        var index = 0
        while (index < destination.size) {
            require(destination[index].isFinite()) { "Non-finite ONNX output" }
            index++
        }
    }

    fun beginProfiling(prefix: File) {
        closeSession()
        profilePrefix = prefix
        try { session = createWithFallback(prefix) }
        catch (exception: Exception) {
            profilePrefix = null
            session = createWithFallback(null)
            throw exception
        }
    }

    fun endProfiling(): File? {
        if (profilePrefix == null) return null
        val path = (session ?: error("Profiling session unavailable")).endProfiling()
        profilePrefix = null
        closeSession(); session = createWithFallback(null)
        return File(path)
    }

    private fun closeSession() {
        val current = session
        session = null
        try { current?.close() } finally { optionsForSession?.close(); optionsForSession = null }
    }

    override fun close() { if (!closed) { closed = true; tensor.close(); closeSession() } }
}
