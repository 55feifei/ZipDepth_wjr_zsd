"""Sequential, same-budget experiments and validation-only model selection."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zipdepth.course.training import train_temporal
from zipdepth.course.evaluation import evaluate_checkpoint
from scripts.export_mobile import export_and_verify
from zipdepth.course.runtime import sha256

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--output',default='output/experiments')
    p.add_argument('--config',default='configs/course_temporal.json')
    p.add_argument('--max-updates',type=int,default=2000)
    p.add_argument('--epochs',type=int,default=5)
    p.add_argument('--gpu-budget-hours',type=float,default=12)
    p.add_argument('--bootstrap',action='store_true',help='Explicitly mark this as a pipeline smoke experiment')
    p.add_argument('--skip-calibration',action='store_true')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    source_manifest=Path(a.manifest).resolve()
    frozen=out/'frozen_manifest.json'
    source_bytes=source_manifest.read_bytes()
    if frozen.exists() and frozen.read_bytes()!=source_bytes:
        raise RuntimeError('Existing experiment manifest differs from current data; use a new output directory or explicitly resume with frozen_manifest.json')
    if not frozen.exists():frozen.write_bytes(source_bytes)
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    cfg.update(manifest=str(frozen.resolve()),epochs=a.epochs,device='cuda',max_updates=a.max_updates)
    train_budget=a.gpu_budget_hours*3600*.75
    per_group=train_budget/3
    started=time.monotonic()
    if not a.skip_calibration:
        cal={**cfg,'output':str(out/'calibration'),'max_updates':min(5,a.max_updates),'max_seconds':min(300,per_group),
             'max_val_pairs':8,'validate_every':5}
        result=train_temporal(cal)
        if not result['updates']:raise RuntimeError('Calibration produced no optimizer update')
        seconds_per_update=result['elapsed_seconds']/result['updates']
        # Leave 15% for checkpointing and periodic validation beyond calibration.
        cfg['max_updates']=min(a.max_updates,max(1,math.floor(per_group*.85/seconds_per_update)))
        (out/'budget.json').write_text(json.dumps({'gpu_budget_hours':a.gpu_budget_hours,'seconds_per_update':seconds_per_update,
          'common_max_updates':cfg['max_updates'],'formal_training_seconds':train_budget,'bootstrap':a.bootstrap},indent=2),encoding='utf-8')
    summaries={};val_results={};baseline_absrel=None
    for weight,name in [(0.,'single'),(.1,'temporal_0.1'),(.3,'temporal_0.3')]:
        run={**cfg,'output':str(out/name),'temporal_weight':weight,'max_seconds':per_group}
        if baseline_absrel is not None:run['baseline_absrel']=baseline_absrel
        resume=out/name/'last.pth'
        if resume.exists():run['resume']=str(resume)
        summaries[name]=train_temporal(run)
        # Sequence-level selection evaluates both saved candidates rather than
        # substituting the trainer's per-pair temporal validation statistic.
        for candidate in ('best','last'):
            ckpt=out/name/f'{candidate}.pth'
            if not ckpt.exists():continue
            tag=f'{name}_{candidate}'
            val_results[tag]=evaluate_checkpoint(frozen,ckpt,'val',out/'validation'/tag,384,'cuda')
        if name=='single':
            single=min((k for k in val_results if k.startswith('single_')),key=lambda k:val_results[k]['absrel'])
            baseline_absrel=val_results[single]['absrel']
    single_key=min((k for k in val_results if k.startswith('single_')),key=lambda k:val_results[k]['absrel'])
    update_counts={k:v['updates'] for k,v in summaries.items()}
    if len(set(update_counts.values()))!=1:
        (out/'incomplete.json').write_text(json.dumps({'reason':'Unequal actual optimizer update counts; no fair comparison or final model published.',
          'updates':update_counts,'summaries':summaries},indent=2),encoding='utf-8')
        raise RuntimeError(f'Budget stopped runs at unequal updates: {update_counts}; resume to equalize before selection')
    eligible=[k for k,v in val_results.items() if k.startswith('temporal_') and v['absrel']<=baseline_absrel*1.02]
    chosen=min([single_key]+eligible,key=lambda k:val_results[k]['temporal_error'])
    single_path=val_results[single_key]['checkpoint'];chosen_path=val_results[chosen]['checkpoint']
    candidate=min((k for k in val_results if k.startswith('temporal_')),key=lambda k:val_results[k]['temporal_error'])
    candidate_path=val_results[candidate]['checkpoint']
    tests={}
    for name,path,smooth in [('original',cfg['checkpoint'],False),('single',single_path,False),
                            ('temporal_candidate',candidate_path,False),('temporal_candidate_smooth',candidate_path,True)]:
        tests[name]=evaluate_checkpoint(frozen,path,'test',out/'test'/name,384,'cuda',smooth)
    tests['selected']=tests['single'] if chosen==single_key else tests['temporal_candidate'] if chosen==candidate else evaluate_checkpoint(frozen,chosen_path,'test',out/'test/selected',384,'cuda')
    single_temporal=tests['single']['temporal_error']
    improvement=(single_temporal-tests['selected']['temporal_error'])/max(single_temporal,1e-12)
    selection={'chosen':chosen,'checkpoint':chosen_path,'single_checkpoint':single_path,'bootstrap':a.bootstrap,
               'temporal_candidate':candidate,'temporal_candidate_checkpoint':candidate_path,
               'eligible_temporal_candidates':eligible,'validation':val_results,'test':tests,
               'temporal_improvement_vs_single':improvement,'target_10_percent_achieved':improvement>=.10,
               'fallback_to_single':chosen==single_key,'total_pipeline_seconds':time.monotonic()-started,
               'actual_updates':update_counts,
               'frozen_manifest_sha256':sha256(frozen),'source_manifest':str(source_manifest),
               'note':'Real small-sample pipeline verification, NOT full course-scale results.' if a.bootstrap else 'Selection uses validation only; test is evaluated after selection.'}
    (out/'selection.json').write_text(json.dumps(selection,indent=2,ensure_ascii=False),encoding='utf-8')
    model_dir=out/'models' if a.bootstrap else Path('output/models')
    export_and_verify(chosen_path,[256,384],model_dir,'zipdepth_smoke' if a.bootstrap else 'zipdepth_final',
                      ['assets/examples/im0.jpg','assets/qualitative/car_show_rgb.jpg'])
    print(json.dumps({k:v for k,v in selection.items() if k not in ('validation','test')},indent=2),flush=True)

if __name__=='__main__':main()
