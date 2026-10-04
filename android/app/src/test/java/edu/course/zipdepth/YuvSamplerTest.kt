package edu.course.zipdepth

import org.junit.Assert.*
import org.junit.Test

class YuvSamplerTest {
    @Test fun supportsPaddedRowsAndInterleavedChroma() {
        val y=YuvPlane(byteArrayOf(16,16,16,16,0,0,16,235.toByte(),235.toByte(),235.toByte(),0,0),6,1)
        val u=YuvPlane(byteArrayOf(128.toByte(),7,128.toByte(),3),4,2)
        val v=YuvPlane(byteArrayOf(128.toByte(),9,128.toByte(),5),4,2)
        assertEquals(0,YuvSampler.rgbAt(y,u,v,0,0))
        assertEquals(0xFFFFFF,YuvSampler.rgbAt(y,u,v,2,1))
        assertEquals(0xFFFFFF,YuvSampler.rgbAt(y,u,v,3,1))
    }

    @Test fun mapsAllClockwiseRotations() {
        assertEquals(3,YuvSampler.sourceIndex(0,1,3,0))
        assertEquals(7,YuvSampler.sourceIndex(0,1,3,90))
        assertEquals(5,YuvSampler.sourceIndex(0,1,3,180))
        assertEquals(1,YuvSampler.sourceIndex(0,1,3,270))
    }
}
