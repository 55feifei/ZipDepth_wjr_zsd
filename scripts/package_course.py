"""Package local deliverables with explicit unfinished experiment/device status."""
import argparse
import json
import sys
import zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zipdepth.course.runtime import sha256

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='output/deliverables/zipdepth-course-progress.zip')
    a=p.parse_args();destination=ROOT/a.output;destination.parent.mkdir(parents=True,exist_ok=True)
    files=set()
    for directory in ['zipdepth','scripts','configs','tests','docs','android']:
        for path in (ROOT/directory).rglob('*'):
            relative=path.relative_to(ROOT).parts
            if path.is_file() and not any(x in relative for x in ['build','.gradle','__pycache__']) and path.name not in ['local.properties']:
                files.add(path)
    for name in ['README.md','README_COURSE.md','TODO.md','requirements.txt','requirements-course.txt','setup.py','LICENSE']:
        files.add(ROOT/name)
    files.add(ROOT/'checkpoints/zipdepth_base_npu.pth')
    for name in ['assets/examples/im0.jpg','assets/examples/clip.mp4','assets/qualitative/car_show_rgb.jpg']:
        files.add(ROOT/name)
    for directory in ['output/models','output/apk','output/videos','output/environment','output/benchmarks','output/tests','output/curves','output/device']:
        if (ROOT/directory).exists():files.update(x for x in (ROOT/directory).rglob('*') if x.is_file())
    selection=ROOT/'output/experiments/selection.json'
    formal=False
    if selection.exists():
        info=json.loads(selection.read_text(encoding='utf-8'))
        formal=not info.get('bootstrap',False)
        files.add(selection)
        for key in ['checkpoint','single_checkpoint','temporal_candidate_checkpoint']:files.add(Path(info[key]).resolve())
        for path in (ROOT/'output/experiments').rglob('*'):
            if path.is_file() and path.suffix in ['.json','.csv','.jsonl'] and 'predictions' not in path.parts:
                files.add(path)
    # Preserve genuine small-sample evidence separately; never label it formal.
    for smoke in (ROOT/'output').glob('smoke*'):
        if (smoke/'selection.json').exists():
            info=json.loads((smoke/'selection.json').read_text(encoding='utf-8'))
            for key in ['checkpoint','single_checkpoint','temporal_candidate_checkpoint']:files.add(Path(info[key]).resolve())
            for path in smoke.rglob('*'):
                if path.is_file() and path.suffix in ['.json','.csv','.jsonl'] and 'predictions' not in path.parts:
                    files.add(path)
    status={'formal_experiments_completed':formal,'phone_verified':False,
            'note':'Current local artifacts. APK build and PC tests do not imply phone installation/benchmark completion.',
            'files':{path.relative_to(ROOT).as_posix():{'bytes':path.stat().st_size,'sha256':sha256(path)} for path in sorted(files)}}
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for path in sorted(files):archive.write(path,path.relative_to(ROOT).as_posix())
        archive.writestr('DELIVERY_STATUS.json',json.dumps(status,ensure_ascii=False,indent=2))
    destination.with_suffix('.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Packaged {len(files)} files: {destination}',flush=True)

if __name__=='__main__':main()
