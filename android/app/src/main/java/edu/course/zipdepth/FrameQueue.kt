package edu.course.zipdepth

import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean

internal class FrameSlot(val size: Int, private val pool: SlotPool) {
    val rgb = FloatArray(3 * size * size)
    val depth = FloatArray(size * size)
    private val inPool = AtomicBoolean(true)
    var token = 0
    var startedNs = 0L
    var preprocessMs = 0.0
    var inferenceMs = 0.0
    var recordable = false

    fun release() = pool.release(this)

    internal fun markCheckedOut() {
        check(inPool.compareAndSet(true, false)) { "slot already checked out" }
    }

    internal fun markReturned() {
        check(inPool.compareAndSet(false, true)) { "slot released twice" }
    }
}

/** Fixed-size buffers for one model resolution. Released slots always return here. */
internal class SlotPool(val size: Int, val epoch: Int) {
    private val free = ArrayBlockingQueue<FrameSlot>(CAPACITY)

    init {
        repeat(CAPACITY) { free.add(FrameSlot(size, this)) }
    }

    fun acquire(): FrameSlot? {
        val slot = free.poll() ?: return null
        slot.markCheckedOut()
        return slot
    }

    fun release(slot: FrameSlot) {
        slot.markReturned()
        check(free.offer(slot)) { "slot pool overflow" }
    }

    private companion object {
        const val CAPACITY = 5
    }
}

/** Capacity-1 handoff that keeps the newest frame and returns the displaced one to its pool. */
internal class LatestHandoff {
    private val queue = ArrayBlockingQueue<FrameSlot>(1)

    fun submitLatest(slot: FrameSlot) {
        while (!queue.offer(slot)) {
            val dropped = queue.poll() ?: continue
            dropped.release()
        }
    }

    fun poll(): FrameSlot? = queue.poll()
}
