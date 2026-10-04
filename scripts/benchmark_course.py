"""PC CPU/GPU latency measurements, recorded separately from phone results."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from zipdepth.course.runtime import DepthRuntime,timing_summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='checkpoints/zipdepth_base_npu.pth')
    p.add_argument('--sizes',type=int,nargs='+',default=[256,384])
    p.add_argument('--devices',nargs='+',default=['cpu','cuda'])
    p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--iterations',type=int,default=200)
    p.add_argument('--output',default='output/benchmarks/pc.json')
    a=p.parse_args();torch.set_num_threads(4)
    if str(a.checkpoint).endswith('.onnx'):
        p.error('This comparison uses PyTorch .pth models; ONNX CPU measurements are recorded by demo_video.py')
    frame=cv2.imdecode(np.fromfile('assets/examples/im0.jpg',np.uint8),cv2.IMREAD_COLOR)
    results=[]
    for device in a.devices:
        for size in a.sizes:
            runtime=DepthRuntime(a.checkpoint,size,device)
            for _ in range(a.warmup):runtime.predict(frame)
            values=[];inference=[];pre=[]
            for _ in range(a.iterations):
                start=time.perf_counter();_,_,timing=runtime.predict(frame)
                values.append((time.perf_counter()-start)*1000)
                inference.append(timing['inference_ms']);pre.append(timing['preprocess_ms'])
            item={'device':device,'size':size,'precision':'float32','backend':'PyTorch fused',
                  'end_to_end':timing_summary(values),'inference':timing_summary(inference),'preprocess':timing_summary(pre)}
            results.append(item);print(json.dumps(item),flush=True)
    info={'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          'torch':torch.__version__,'warmup':a.warmup,'results':results,
          'note':'PC results only; includes resize, model, transfer to CPU; excludes camera acquisition and display.'}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(info,indent=2),encoding='utf-8')

if __name__=='__main__':main()
