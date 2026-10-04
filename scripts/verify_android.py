"""Install/collect real Android artifacts; never synthesize phone results."""
import argparse
import csv
import io
import json
import math
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zipdepth.course.runtime import sha256

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'edu.course.zipdepth'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adb', default='output/tools/android/sdk/platform-tools/adb.exe')
    parser.add_argument('--serial')
    parser.add_argument('--apk', default='output/apk/zipdepth-baseline-debug.apk')
    parser.add_argument('--install', action='store_true')
    parser.add_argument('--collect', action='store_true', help='Collect measurements produced by the in-app five-minute tests')
    parser.add_argument('--record', action='store_true', help='Record the actual foreground phone screen')
    parser.add_argument('--record-seconds', type=int, default=120)
    parser.add_argument('--output', default='output/device')
    args = parser.parse_args()
    adb = (ROOT / args.adb).resolve()
    if not adb.is_file():
        raise FileNotFoundError(f'ADB missing: {adb}; run android/setup-build.ps1')
    destination = ROOT / args.output
    destination.mkdir(parents=True, exist_ok=True)
    listing = subprocess.run([str(adb), 'devices'], check=True, capture_output=True, text=True).stdout
    devices = [line.split()[0] for line in listing.splitlines()[1:] if len(line.split()) >= 2 and line.split()[1] == 'device']
    if not devices:
        status = {'phone_verified': False, 'status': 'no_authorized_device',
                  'action': 'Connect Honor 100 Pro with USB debugging and authorize the computer.'}
        (destination / 'acceptance.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(json.dumps(status, indent=2))
        return
    if args.serial:
        if args.serial not in devices:
            raise ValueError('Selected serial is not an authorized device')
        serial = args.serial
    elif len(devices) == 1:
        serial = devices[0]
    else:
        raise ValueError('Multiple phones connected; pass --serial to select the course device')
    prefix = [str(adb), '-s', serial]

    def run(*command, binary=False):
        result = subprocess.run(prefix + list(command), check=True, capture_output=True)
        return result.stdout if binary else result.stdout.decode('utf-8', errors='replace').strip()

    properties = {key: run('shell', 'getprop', prop) for key, prop in {
        'manufacturer': 'ro.product.manufacturer', 'model': 'ro.product.model',
        'android_release': 'ro.build.version.release', 'api': 'ro.build.version.sdk',
        'soc': 'ro.soc.model', 'abi': 'ro.product.cpu.abi'}.items()}
    properties['collected_unix'] = time.time()
    (destination / 'device.json').write_text(json.dumps(properties, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.install:
        apk = (ROOT / args.apk).resolve()
        if not apk.is_file():
            raise FileNotFoundError(apk)
        install_result = run('install', '-r', str(apk))
        run('shell', 'am', 'start', '-n', f'{PACKAGE}/.MainActivity')
        (destination / 'installation.json').write_text(json.dumps({
            'apk': str(apk), 'sha256': sha256(apk), 'result': install_result,
            'note': 'Installation and launch alone do not verify camera permissions or sustained FPS.'}, indent=2), encoding='utf-8')
    if args.record:
        if not 1 <= args.record_seconds <= 180:
            raise ValueError('Screen recording duration must be 1..180 seconds')
        # Only this explicitly named course video is written on the device.
        remote = '/sdcard/zipdepth-course-demo.mp4'
        run('shell', 'screenrecord', '--time-limit', str(args.record_seconds), remote)
        run('pull', remote, str(destination / 'demo.mp4'))
    if args.collect:
        data = run('exec-out', 'run-as', PACKAGE, 'tar', '-cf', '-', 'files/measurements', binary=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
            for member in archive.getmembers():
                parts = PurePosixPath(member.name).parts
                if PurePosixPath(member.name).is_absolute() or '..' in parts or member.issym() or member.islnk():
                    raise ValueError(f'Unsafe phone archive member: {member.name}')
            archive.extractall(destination, filter='data')
    measurements = []
    for path in destination.glob('files/measurements/*/summary.json'):
        summary = json.loads(path.read_text(encoding='utf-8'))
        csv_path = path.with_name('frames.csv')
        with csv_path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        size = int(summary['resolution'])
        flavor = summary.get('model_flavor', 'zipdepth_baseline').removeprefix('zipdepth_')
        local_model = ROOT / f'output/models/zipdepth_{flavor}_{size}.onnx'
        reasons = []
        if not summary.get('completed'):
            reasons.append('phone_test_interrupted')
        if summary.get('observed_update_span_s', 0) < 299:
            reasons.append('less_than_five_minutes')
        if not rows or len(rows) != summary.get('depth_updates'):
            reasons.append('frame_count_mismatch')
        if not local_model.exists() or sha256(local_model) != summary.get('model_sha256'):
            reasons.append('model_provenance_mismatch')
        stages = {}
        for column in ['preprocess_ms','inference_ms','postprocess_ms','ui_queue_ms','pipeline_ms']:
            values = sorted(float(row[column]) for row in rows)
            if not values or not all(math.isfinite(value) and value >= 0 for value in values):
                reasons.append(f'invalid_{column}')
                continue
            def percentile(q):
                index = (len(values)-1)*q
                low = int(index)
                return values[low] + (index-low)*(values[min(low+1,len(values)-1)]-values[low])
            stages[column] = {'mean': sum(values)/len(values), 'p50': percentile(.5), 'p95': percentile(.95)}
        measurements.append({'summary': str(path.relative_to(destination)), 'size': size,
                             'model_flavor': flavor, 'model_sha256': summary.get('model_sha256'),
                             'valid_continuous_measurement': not reasons, 'invalid_reasons': reasons,
                             'fps': summary.get('display_update_fps'),
                             'stages_ms': stages,
                             'p50_ms': summary.get('pipeline_p50_ms'), 'p95_ms': summary.get('pipeline_p95_ms')})
    valid = [m for m in measurements if m['valid_continuous_measurement']]
    # Use the most recent valid measurement for each resolution and model.
    latest = {}
    for measure in sorted(valid, key=lambda m: m['summary']):
        latest[(measure['model_flavor'], measure['size'])] = measure
    final = [m for (flavor, _), m in latest.items() if flavor == 'final']
    tested_sizes = sorted({m['size'] for m in final})
    passing = [m['size'] for m in final if m['fps'] >= 15]
    status = {'phone_verified': False, 'device': properties, 'measurements': measurements,
              'final_model_sustained_performance_complete': tested_sizes == [256, 384],
              'recommended_final_size': max(passing) if passing else 256,
              'target_15_fps_met': bool(passing),
              'remaining': ['Manually verify permission denial/retry, rotation, background/foreground recovery and demo recording.',
                            'Record acceptance evidence before marking TODO step 12 complete.']}
    (destination / 'acceptance.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
