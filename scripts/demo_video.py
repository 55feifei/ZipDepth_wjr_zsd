"""Produce honest RGB/depth comparison videos from actual model inference."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from zipdepth.course.runtime import DepthRuntime,GatedEMA,StableColorizer,timing_summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',default='assets/examples/clip.mp4')
    p.add_argument('--models',nargs='+',default=['baseline=output/models/zipdepth_baseline_256.onnx'])
    p.add_argument('--smooth-models',nargs='*',default=[])
    p.add_argument('--input-size',type=int,default=256)
    p.add_argument('--device',default='cpu')
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--output',default='output/videos/baseline_demo.mp4')
    a=p.parse_args();cap=cv2.VideoCapture(a.input)
    if not cap.isOpened():raise ValueError(f'Cannot open {a.input}')
    sourcefps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(sourcefps) or sourcefps<=0:sourcefps=30
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    runtimes={};filters={};latencies={};predictions={};rgbs=[]
    for item in a.models:
        name,path=item.split('=',1);runtimes[name]=DepthRuntime(path,a.input_size,a.device)
        filters[name]=GatedEMA();latencies[name]=[];predictions[name]=[]
    count=0
    try:
        while not a.max_frames or count<a.max_frames:
            ok,frame=cap.read()
            if not ok:break
            for name,runtime in runtimes.items():
                start=time.perf_counter();d,rgb,stats=runtime.predict(frame)
                if name in a.smooth_models:d=filters[name].update(rgb,d)
                latencies[name].append((time.perf_counter()-start)*1000)
                predictions[name].append(d)
            rgbs.append((rgb[:,:,::-1]*255).astype(np.uint8));count+=1
            if count%30==0:print(f'Inferred {count} frames',flush=True)
    finally:cap.release()
    if not count:raise ValueError('No decoded video frames')
    # A shared fixed scale makes visual method comparisons meaningful.
    bounds=np.percentile(np.concatenate([np.asarray(v)[:,::8,::8].ravel() for v in predictions.values()]),[2,98])
    color=StableColorizer(fixed_range=bounds)
    width=(len(runtimes)+1)*a.input_size;height=a.input_size+46
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),sourcefps,(width,height))
    if not writer.isOpened():raise RuntimeError('MP4 writer failed')
    try:
        for i,rgb in enumerate(rgbs):
            canvas=np.zeros((height,width,3),np.uint8);canvas[46:,:a.input_size]=rgb
            cv2.putText(canvas,'RGB',(8,27),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1,cv2.LINE_AA)
            for j,(name,depths) in enumerate(predictions.items(),1):
                x=j*a.input_size;canvas[46:,x:x+a.input_size]=color.update(depths[i])
                cv2.putText(canvas,name,(x+8,22),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
                cv2.putText(canvas,'relative inverse depth',(x+8,39),cv2.FONT_HERSHEY_SIMPLEX,.35,(180,180,180),1,cv2.LINE_AA)
            writer.write(canvas)
            if i==min(30,count-1):cv2.imencode('.jpg',canvas)[1].tofile(str(output.with_suffix('.jpg')))
    finally:writer.release()
    info={'input':a.input,'frames':count,'playback_fps':sourcefps,'models':a.models,
          'color_range':bounds.tolist(),'latency':{n:timing_summary(v[3:] or v) for n,v in latencies.items()},
          'note':'Playback FPS is source video FPS, not measured inference throughput. Latency excludes video decode/encode.'}
    output.with_suffix('.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__':main()
