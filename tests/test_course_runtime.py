import json
import unittest
from pathlib import Path
import numpy as np
from zipdepth.course.runtime import GatedEMA,preprocess_bgr
from zipdepth.course.evaluation import flow_error_change,spatial_metrics

class RuntimeTests(unittest.TestCase):
    def test_fixture(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/smoothing.json').read_text())
        for case in fixture['cases']:
            ema=GatedEMA();h,w=case['height'],case['width']
            for frame in case['frames']:
                rgb=np.array(frame['rgb'],np.float32).reshape(3,h,w).transpose(1,2,0)
                d=np.array(frame['depth'],np.float32).reshape(h,w)
                np.testing.assert_allclose(ema.update(rgb,d),np.array(frame['expected']).reshape(h,w),atol=1e-6)
                self.assertEqual(ema.scene_reset,frame['scene_reset'])
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):GatedEMA().update(np.zeros((2,2,3)),np.full((2,2),np.nan))
    def test_center_crop_rgb(self):
        bgr=np.zeros((4,8,3),np.uint8);bgr[:,2:6,2]=255
        x,rgb=preprocess_bgr(bgr,4)
        self.assertEqual(x.shape,(1,3,4,4));self.assertTrue((rgb[:,:,0]==1).all());self.assertTrue((rgb[:,:,2]==0).all())
    def test_real_change_not_penalized(self):
        t0=np.ones((3,4),np.float32);t1=t0*2
        mask=np.ones_like(t0,bool);flow=np.zeros((2,3,4),np.float32)
        err,n=flow_error_change(t0,t1,t0,t1,mask,mask,flow,mask,1,0)
        self.assertEqual(err,0);self.assertEqual(n,12)
    def test_flow_direction(self):
        d=np.arange(12,dtype=np.float32).reshape(3,4)/10+1
        latter=np.zeros_like(d);latter[:,1:]=d[:,:-1]
        flow=np.zeros((2,3,4),np.float32);flow[0]=1
        m=np.ones_like(d,bool)
        err,n=flow_error_change(d,latter,np.zeros_like(d),np.zeros_like(d),m,m,flow,m,1,0)
        self.assertEqual(err,0);self.assertEqual(n,9)
    def test_spatial_alignment(self):
        t=np.arange(1,13,dtype=np.float32).reshape(3,4)/20
        result=spatial_metrics(t*2+.1,t,np.ones_like(t,bool))
        self.assertLess(result['absrel'],1e-6);self.assertEqual(result['delta1'],1.)
    def test_subpixel_occluded_neighbor_rejected(self):
        d=np.ones((2,3),np.float32);m=np.ones_like(d,bool)
        next_mask=m.copy();next_mask[:,1]=False
        flow=np.zeros((2,2,3),np.float32);flow[0]=.01
        first_mask=np.zeros_like(m);first_mask[:,0]=True
        _,n=flow_error_change(d,d,d,d,first_mask,next_mask,flow,m,1,0)
        self.assertEqual(n,0)

if __name__=='__main__':unittest.main()
