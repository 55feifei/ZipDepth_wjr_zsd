"""Train mobile ZipDepth on a TartanAir pair manifest using the ML1 runtime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zipdepth.course.training import train_temporal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/course_temporal.json')
    parser.add_argument('--manifest')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output')
    parser.add_argument('--temporal-weight', type=float)
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--resume')
    parser.add_argument('--input-size', type=int)
    parser.add_argument('--device')
    parser.add_argument('--grad-accum', type=int)
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--max-val-pairs', type=int)
    parser.add_argument('--validate-every', type=int)
    parser.add_argument('--baseline-absrel', type=float)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    config.update({name: value for name, value in vars(args).items() if name != 'config' and value is not None})
    for required in ('manifest', 'checkpoint', 'output'):
        if not config.get(required):
            parser.error(f'--{required} is required')
    print(json.dumps(train_temporal(config), indent=2))


if __name__ == '__main__':
    main()
