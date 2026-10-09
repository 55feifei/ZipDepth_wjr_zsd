package edu.course.zipdepth

import org.junit.Assert.assertEquals
import org.junit.Test

class InputSizeTest {
    @Test fun unsetSizeDefaultsTo384() {
        assertEquals(384, InputSize.resolve(null))
        assertEquals(1, InputSize.spinnerIndex(InputSize.resolve(null)))
    }

    @Test fun saved384StaysSelected() {
        assertEquals(384, InputSize.resolve(384))
        assertEquals(1, InputSize.spinnerIndex(384))
    }

    @Test fun saved256IsRestored() {
        assertEquals(256, InputSize.resolve(256))
        assertEquals(0, InputSize.spinnerIndex(256))
    }

    @Test fun unsupportedStoredSizeFallsBackTo384() {
        assertEquals(384, InputSize.resolve(128))
        assertEquals(1, InputSize.spinnerIndex(128))
    }
}
