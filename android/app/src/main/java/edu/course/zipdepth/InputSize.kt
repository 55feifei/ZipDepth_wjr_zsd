package edu.course.zipdepth

/** First launch uses 384. A later launch restores only an explicit 256 or 384 choice. */
object InputSize {
    const val DEFAULT = 384

    fun resolve(stored: Int?): Int = when (stored) {
        256, 384 -> stored
        else -> DEFAULT
    }

    fun spinnerIndex(size: Int): Int = if (resolve(size) == 384) 1 else 0
}
