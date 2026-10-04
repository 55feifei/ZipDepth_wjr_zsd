import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from zipdepth.course.evaluation import evaluate_checkpoint

def main():
    p=argparse.ArgumentParser(description='Sequence-aligned temporal and spatial evaluation')
    p.add_argument('--manifest',required=True)
    p.add_argument('--models',nargs='+',required=True,help='name=checkpoint entries')
    p.add_argument('--smooth-models',nargs='*',default=[])
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--input-size',type=int,default=384)
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',default='output/evaluation')
    a=p.parse_args();results={}
    for item in a.models:
        name,checkpoint=item.split('=',1)
        if Path(name).name!=name: p.error('Model names must be simple directory names')
        results[name]=evaluate_checkpoint(a.manifest,checkpoint,a.split,Path(a.output)/a.split/name,
                                        a.input_size,a.device,name in a.smooth_models)
    path=Path(a.output)/a.split/'comparison.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')

if __name__=='__main__':main()
