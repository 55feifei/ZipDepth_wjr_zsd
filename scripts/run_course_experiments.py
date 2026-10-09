"""Sequential, same-budget experiments and validation-only model selection."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zipdepth.course.training import measure_subset_validation, train_temporal
from zipdepth.course.evaluation import evaluate_checkpoint
from scripts.export_mobile import export_and_verify
from zipdepth.course.runtime import sha256

PRIMARY_WEIGHTS = [(0.0, 'single'), (0.1, 'temporal_0.1'), (0.3, 'temporal_0.3')]


def _read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _add_unique(found, path):
    path = Path(path)
    if not path.is_file():
        return
    resolved = str(path.resolve())
    if resolved not in {str(Path(item).resolve()) for item in found}:
        found.append(str(path))


def subset_candidates(group_dir, baseline_subset, single):
    """Keep last, best, and the strongest same-subset snapshots for full validation."""
    records = [row for row in _read_jsonl(group_dir / 'subset_validation.jsonl') if row.get('snapshot')]
    found = []
    _add_unique(found, group_dir / 'last.pth')
    _add_unique(found, group_dir / 'best.pth')
    if not records:
        return found
    if single or baseline_subset is None:
        _add_unique(found, group_dir / min(records, key=lambda row: row['absrel'])['snapshot'])
        _add_unique(found, group_dir / min(records, key=lambda row: row['spatial'])['snapshot'])
        return found
    eligible = [row for row in records if row['absrel'] <= baseline_subset * 1.02]
    if eligible:
        _add_unique(found, group_dir / min(eligible, key=lambda row: row['temporal'])['snapshot'])
        _add_unique(found, group_dir / min(eligible, key=lambda row: row['absrel'])['snapshot'])
    return found


def _tag(group, path):
    stem = Path(path).stem
    return f'{group}_{stem}'


def train_one(cfg, out, name, weight, per_group, baseline_absrel):
    run = {**cfg, 'output': str(out / name), 'temporal_weight': weight, 'max_seconds': per_group}
    if baseline_absrel is not None:
        run['baseline_absrel'] = baseline_absrel
    resume = out / name / 'last.pth'
    if resume.exists():
        run['resume'] = str(resume)
    return train_temporal(run)


def full_validate(frozen, out, group, paths, val_results):
    for path in paths:
        tag = _tag(group, path)
        if tag in val_results:
            continue
        print(f'full validation {tag}', flush=True)
        val_results[tag] = evaluate_checkpoint(frozen, path, 'val', out / 'validation' / tag, 384, 'cuda')


def select_and_report(a, out, cfg, frozen, source_manifest, started, summaries, val_results, subset_baseline):
    single_keys = [key for key in val_results if key.startswith('single_')]
    if not single_keys:
        raise RuntimeError('No single-frame checkpoint reached full validation')
    single_key = min(single_keys, key=lambda key: val_results[key]['absrel'])
    baseline_absrel = val_results[single_key]['absrel']
    update_counts = {key: value['updates'] for key, value in summaries.items()}
    if len(set(update_counts.values())) != 1:
        (out / 'incomplete.json').write_text(json.dumps({
            'reason': 'Unequal actual optimizer update counts; no fair comparison or final model published.',
            'updates': update_counts, 'summaries': summaries}, indent=2), encoding='utf-8')
        raise RuntimeError(f'Budget stopped runs at unequal updates: {update_counts}; resume to equalize before selection')
    eligible = [key for key, value in val_results.items()
                if key.startswith('temporal_') and value['absrel'] <= baseline_absrel * 1.02]
    chosen = min([single_key] + eligible, key=lambda key: val_results[key]['temporal_error'])
    temporal_keys = [key for key in val_results if key.startswith('temporal_')]
    candidate = min(temporal_keys, key=lambda key: val_results[key]['temporal_error']) if temporal_keys else chosen
    single_path = val_results[single_key]['checkpoint']
    chosen_path = val_results[chosen]['checkpoint']
    candidate_path = val_results[candidate]['checkpoint']
    selection = {
        'chosen': chosen, 'checkpoint': chosen_path, 'single_checkpoint': single_path,
        'bootstrap': a.bootstrap, 'temporal_candidate': candidate,
        'temporal_candidate_checkpoint': candidate_path,
        'eligible_temporal_candidates': eligible, 'validation': val_results,
        'subset_baseline': subset_baseline,
        'full_validation_absrel_gate': baseline_absrel * 1.02,
        'fallback_to_single': chosen == single_key,
        'total_pipeline_seconds': time.monotonic() - started,
        'actual_updates': update_counts,
        'frozen_manifest_sha256': sha256(frozen), 'source_manifest': str(source_manifest),
        'note': 'Choice uses full validation only. Subset metrics only decided which checkpoints to keep and which few to fully evaluate. Test metrics, when present, do not change the choice.',
    }
    (out / 'selection.json').write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding='utf-8')
    tests = {}
    for name, path, smooth in [('original', cfg['checkpoint'], False), ('single', single_path, False),
                               ('temporal_candidate', candidate_path, False),
                               ('temporal_candidate_smooth', candidate_path, True)]:
        print(f'test evaluation {name}', flush=True)
        tests[name] = evaluate_checkpoint(frozen, path, 'test', out / 'test' / name, 384, 'cuda', smooth)
    tests['selected'] = tests['single'] if chosen == single_key else tests['temporal_candidate'] if chosen == candidate else evaluate_checkpoint(
        frozen, chosen_path, 'test', out / 'test' / 'selected', 384, 'cuda')
    single_temporal = tests['single']['temporal_error']
    improvement = (single_temporal - tests['selected']['temporal_error']) / max(single_temporal, 1e-12)
    selection['test'] = tests
    selection['temporal_improvement_vs_single'] = improvement
    selection['target_10_percent_achieved'] = improvement >= 0.10
    selection['total_pipeline_seconds'] = time.monotonic() - started
    (out / 'selection.json').write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding='utf-8')
    export_and_verify(chosen_path, [256, 384], out / 'models', 'zipdepth_selected',
                      ['assets/examples/im0.jpg', 'assets/qualitative/car_show_rgb.jpg'])
    print(json.dumps({key: value for key, value in selection.items() if key not in ('validation', 'test')}, indent=2), flush=True)


def run_fixed_updates(a, out, cfg, per_group, frozen, source_manifest, started):
    """Train equal update counts, gate best.pth on one subset, then fully validate a few candidates."""
    summaries = {}
    val_results = {}
    summary = train_one(cfg, out, 'single', 0.0, per_group, None)
    summaries['single'] = summary
    reference = out / 'single' / 'best.pth'
    if not reference.is_file():
        reference = out / 'single' / 'last.pth'
    subset_baseline = measure_subset_validation(reference, cfg)
    (out / 'subset_baseline.json').write_text(json.dumps(subset_baseline, indent=2), encoding='utf-8')
    print(f"subset baseline absrel={subset_baseline['absrel']:.6f} pairs={subset_baseline['pairs']}", flush=True)
    for weight, name in PRIMARY_WEIGHTS[1:]:
        summaries[name] = train_one(cfg, out, name, weight, per_group, subset_baseline['absrel'])
    for weight, name in PRIMARY_WEIGHTS:
        full_validate(frozen, out, name, subset_candidates(out / name, subset_baseline['absrel'], name == 'single'), val_results)
    single_absrel = min(value['absrel'] for key, value in val_results.items() if key.startswith('single_'))
    gate = single_absrel * 1.02
    temporal_03 = [value for key, value in val_results.items() if key.startswith('temporal_0.3')]
    temporal_01 = [value for key, value in val_results.items() if key.startswith('temporal_0.1')]
    if temporal_03 and min(value['absrel'] for value in temporal_03) > gate:
        extra_weight = 0.15 if temporal_01 and min(value['absrel'] for value in temporal_01) <= gate else 0.05
        extra_name = f'temporal_{extra_weight:.2f}'
        print(f'{extra_name} follows because temporal_0.3 exceeds the full-validation gate {gate:.6f}', flush=True)
        summaries[extra_name] = train_one(cfg, out, extra_name, extra_weight, per_group, subset_baseline['absrel'])
        full_validate(frozen, out, extra_name, subset_candidates(out / extra_name, subset_baseline['absrel'], False), val_results)
    select_and_report(a, out, cfg, frozen, source_manifest, started, summaries, val_results, subset_baseline)

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
    p.add_argument('--fixed-updates',action='store_true',help='Train every group to --max-updates; do not shrink the count from the time budget')
    p.add_argument('--checkpoint-every',type=int,default=0)
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
        if a.fixed_updates:
            cfg['max_updates']=a.max_updates
            per_group=max(per_group,a.max_updates*max(seconds_per_update,1.0)*2)
        (out/'budget.json').write_text(json.dumps({'gpu_budget_hours':a.gpu_budget_hours,'seconds_per_update':seconds_per_update,
          'common_max_updates':cfg['max_updates'],'formal_training_seconds':train_budget,'per_group_seconds':per_group,
          'fixed_updates':a.fixed_updates,'bootstrap':a.bootstrap},indent=2),encoding='utf-8')
    if a.fixed_updates:
        cfg['checkpoint_every']=a.checkpoint_every or 500
        run_fixed_updates(a,out,cfg,per_group,frozen,source_manifest,started)
        return
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
