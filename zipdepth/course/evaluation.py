"""Spatial metrics plus sequence-aligned, flow-compensated error changes.

Predictions are relative inverse depth. Sequence alignment is fit ONCE per
trajectory, never independently per pair for the temporal measurement.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from zipdepth.course.runtime import GatedEMA, load_model, sha256
from zipdepth.course.losses import flow_warp_next


def fit_affine(pred, target, mask):
    p, t = np.asarray(pred)[mask].astype(np.float64), np.asarray(target)[mask].astype(np.float64)
    if p.size < 2:
        raise ValueError('Too few valid alignment pixels')
    a, b = np.linalg.lstsq(np.stack([p, np.ones_like(p)], axis=1), t, rcond=None)[0]
    return float(a), float(b)


def spatial_metrics(pred, target, mask, max_depth=80.):
    a, b = fit_affine(pred, target, mask)
    gt = 1. / target[mask]
    estimated = 1. / np.maximum(a*pred[mask]+b, 1/max_depth)
    ratio = np.maximum(estimated / gt, gt / estimated)
    return {'absrel':float(np.mean(np.abs(estimated-gt)/gt)),
            'rmse':float(np.sqrt(np.mean((estimated-gt)**2))),
            'delta1':float(np.mean(ratio < 1.25)), 'alignment_scale':a,'alignment_shift':b,
            'valid_pixels':int(mask.sum())}


def flow_error_change(pred0, pred1, target0, target1, mask0, mask1, flow, flow_mask, scale, shift):
    e0=scale*pred0+shift-target0
    e1=scale*pred1+shift-target1
    # Use the SAME exact bilinear sampler as training. OpenCV INTER_LINEAR
    # quantizes subpixel maps to 1/32, incorrectly accepting occluded neighbors.
    packed=torch.from_numpy(np.stack([e1,mask1.astype(np.float32)]).astype(np.float32))[None]
    warped,inside=flow_warp_next(packed,torch.from_numpy(np.asarray(flow,np.float32))[None])
    warped=warped[0].numpy()
    valid=mask0 & (warped[1]>=1-1e-6) & flow_mask & inside[0,0].numpy()
    if not valid.any():return 0.,0
    return float(np.abs(e0-warped[0])[valid].sum()),int(valid.sum())


@torch.inference_mode()
def evaluate_checkpoint(manifest_path, checkpoint, split, output_dir, size=384, device='cuda', smoothing=False):
    from zipdepth.course.data import TartanAirPairDataset
    dataset=TartanAirPairDataset(manifest_path,split=split,input_size=size,augment=False)
    if not len(dataset):raise ValueError(f'Empty split: {split}')
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    cache=output/'predictions';cache.mkdir(exist_ok=True)
    model=load_model(checkpoint,device,fused=True)
    totals={};records=[];seen=set();ema={};last_ids={};pair_ids=[]
    # Sort by trajectory/time for causal smoothing. A dataset index is retained
    # so the loader remains the source of crop/flow/mask truth in the second pass.
    pairs=dataset.pairs
    order=sorted(range(len(dataset)),key=lambda i:(pairs[i]['trajectory'],pairs[i]['frame_id']))
    for index in order:
        sample=dataset[index]
        traj=str(sample['trajectory']);fid=int(sample['frame_id']);pair_ids.append(index)
        row={k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in sample.items()}
        images=row['image'];targets=row['target'][:,0];masks=row['mask'][:,0].astype(bool)
        for j in range(2):
            frameid=fid+j
            key=(traj,frameid)
            if key in seen:continue
            seen.add(key)
            pred=model(torch.from_numpy(images[j:j+1]).to(device))[0,0].float().cpu().numpy()
            if not np.isfinite(pred).all():raise ValueError(f'Nonfinite output: {key}')
            if smoothing:
                filter_=ema.setdefault(traj,GatedEMA())
                if last_ids.get(traj,frameid-1)!=frameid-1:filter_.reset()
                pred=filter_.update(images[j].transpose(1,2,0),pred)
                last_ids[traj]=frameid
            token=traj.replace('/','__').replace('\\','__')+f'__{frameid:06d}'
            np.save(cache/f'{token}.npy',pred)
            metric=spatial_metrics(pred,targets[j],masks[j])
            records.append({'trajectory':traj,'frame_id':frameid,**metric})
            # Deterministic strided pixels keep sums bounded without keeping
            # entire video tensors in RAM. Same sampling is used for every model.
            valid=masks[j][::4,::4]
            p=pred[::4,::4][valid].astype(np.float64);t=targets[j][::4,::4][valid].astype(np.float64)
            v=totals.setdefault(traj,np.zeros(6,np.float64))
            v+=np.array([len(p),p.sum(),(p*p).sum(),t.sum(),(p*t).sum(),(t*t).sum()])
        if len(pair_ids)%100==0:print(f'{split}: {len(pair_ids)}/{len(dataset)} pairs',flush=True)
    alignment={}
    for traj,(n,sp,spp,st,spt,stt) in totals.items():
        a,b=np.linalg.lstsq(np.array([[spp,sp],[sp,n]]),np.array([spt,st]),rcond=None)[0]
        alignment[traj]={'scale':float(a),'shift':float(b),'mean_inverse_depth':float(st/max(n,1))}
    temporal={traj:[0.,0] for traj in totals}
    for index in pair_ids:
        sample=dataset[index];traj=str(sample['trajectory']);fid=int(sample['frame_id'])
        token=traj.replace('/','__').replace('\\','__')
        p0=np.load(cache/f'{token}__{fid:06d}.npy');p1=np.load(cache/f'{token}__{fid+1:06d}.npy')
        t=sample['target'].numpy()[:,0];mask=sample['mask'].numpy()[:,0].astype(bool)
        al=alignment[traj]
        err,n=flow_error_change(p0,p1,t[0],t[1],mask[0],mask[1],sample['flow'].numpy(),
                                sample['flow_mask'].numpy()[0].astype(bool),al['scale'],al['shift'])
        temporal[traj][0]+=err;temporal[traj][1]+=n
    sequences=[]
    for traj in totals:
        al=alignment[traj];rows=[r for r in records if r['trajectory']==traj]
        err,n=temporal[traj]
        normalized=err/max(n,1)/max(al['mean_inverse_depth'],1e-8)
        scales=np.array([r['alignment_scale'] for r in rows])
        shifts=np.array([r['alignment_shift'] for r in rows])
        sequences.append({'trajectory':traj,'frames':len(rows),'valid_flow_pixels':n,
                          'temporal_error':normalized if n else None,
                          'scale_drift':float(np.std(scales)/max(abs(al['scale']),1e-8)),
                          'shift_drift':float(np.std(shifts)/max(al['mean_inverse_depth'],1e-8)),**al})
    valid_seq=[s for s in sequences if s['valid_flow_pixels']]
    if not valid_seq:raise ValueError('No valid temporal evaluation pixels')
    summary={'checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'split':split,'size':size,
             'smoothing':smoothing,'frames':len(records),'pairs':len(dataset),
             **{k:float(np.mean([r[k] for r in records])) for k in ['absrel','rmse','delta1']},
             'temporal_error':float(np.mean([s['temporal_error'] for s in valid_seq])),
             'scale_drift':float(np.mean([s['scale_drift'] for s in sequences])),
             'shift_drift':float(np.mean([s['shift_drift'] for s in sequences])),
             'alignment_policy':'spatial:per-frame inverse-depth affine; temporal:one affine per full trajectory',
             'sequences':sequences}
    (output/'metrics.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    with (output/'per_frame.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    print(json.dumps({k:v for k,v in summary.items() if k!='sequences'},indent=2),flush=True)
    return summary
