"""Install/collect real Android artifacts; never synthesize phone results."""
import argparse
import csv
import io
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zipdepth.course.android_acceptance import assess_measurement, build_acceptance, manual_is_complete
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
    properties = None
    device_connected = bool(devices)
    if not devices:
        print('No authorized phone. Local measurement files will still be summarized; nothing is invented.')
    elif args.serial:
        if args.serial not in devices:
            raise ValueError('Selected serial is not an authorized device')
        serial = args.serial
    elif len(devices) == 1:
        serial = devices[0]
    else:
        raise ValueError('Multiple phones connected; pass --serial to select the course device')
    if device_connected:
        prefix = [str(adb), '-s', serial]

    def run(*command, binary=False):
        result = subprocess.run(prefix + list(command), check=True, capture_output=True)
        return result.stdout if binary else result.stdout.decode('utf-8', errors='replace').strip()

    if device_connected:
        properties = {key: run('shell', 'getprop', prop) for key, prop in {
        'manufacturer': 'ro.product.manufacturer', 'model': 'ro.product.model',
        'android_release': 'ro.build.version.release', 'api': 'ro.build.version.sdk',
        'soc': 'ro.soc.model', 'abi': 'ro.product.cpu.abi'}.items()}
        properties['collected_unix'] = time.time()
        (destination / 'device.json').write_text(json.dumps(properties, ensure_ascii=False, indent=2), encoding='utf-8')
    if device_connected and args.install:
        apk = (ROOT / args.apk).resolve()
        if not apk.is_file():
            raise FileNotFoundError(apk)
        install_result = run('install', '-r', str(apk))
        run('shell', 'am', 'start', '-n', f'{PACKAGE}/.MainActivity')
        (destination / 'installation.json').write_text(json.dumps({
            'apk': str(apk), 'sha256': sha256(apk), 'result': install_result,
            'note': 'Installation and launch alone do not verify camera permissions or sustained FPS.'}, indent=2), encoding='utf-8')
    if device_connected and args.record:
        if not 1 <= args.record_seconds <= 180:
            raise ValueError('Screen recording duration must be 1..180 seconds')
        # Only this explicitly named course video is written on the device.
        remote = '/sdcard/zipdepth-course-demo.mp4'
        run('shell', 'screenrecord', '--time-limit', str(args.record_seconds), remote)
        run('pull', remote, str(destination / 'demo.mp4'))
    if device_connected and args.collect:
        data = run('exec-out', 'run-as', PACKAGE, 'tar', '-cf', '-', 'files/measurements', binary=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as archive:
            for member in archive.getmembers():
                parts = PurePosixPath(member.name).parts
                if PurePosixPath(member.name).is_absolute() or '..' in parts or member.issym() or member.islnk():
                    raise ValueError(f'Unsafe phone archive member: {member.name}')
            archive.extractall(destination, filter='data')
    # Raw CSV, JSON and profiling files remain under files/measurements. acceptance.json is only a summary.
    measurements = []
    for path in sorted(destination.glob('files/measurements/*/summary.json')):
        summary = json.loads(path.read_text(encoding='utf-8'))
        csv_path = path.with_name('frames.csv')
        if csv_path.is_file():
            with csv_path.open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
        else:
            rows = None
        try:
            size = int(summary['resolution'])
        except (KeyError, TypeError, ValueError):
            size = None
        flavor = str(summary.get('model_flavor', 'zipdepth_baseline'))
        short_flavor = flavor.removeprefix('zipdepth_')
        local_model = None if size is None else next((
            candidate for candidate in (
                ROOT / 'output' / 'experiments_5600' / 'models' / f'zipdepth_{short_flavor}_{size}.onnx',
                ROOT / 'output' / 'models' / f'zipdepth_{short_flavor}_{size}.onnx',
            ) if candidate.is_file()
        ), None)
        matches = bool(local_model and local_model.is_file() and sha256(local_model) == summary.get('model_sha256'))
        assessed = assess_measurement(summary, rows, matches)
        assessed['summary'] = str(path.relative_to(destination))
        measurements.append(assessed)
    manual_path = destination / 'manual_acceptance.json'
    manual_document = json.loads(manual_path.read_text(encoding='utf-8')) if manual_path.is_file() else None
    demo = destination / 'demo.mp4'
    status = build_acceptance(
        measurements,
        manual_is_complete(manual_document),
        demo.is_file() and demo.stat().st_size > 0,
        device=properties,
        device_connected=device_connected,
    )
    if not device_connected:
        status['status'] = 'no_authorized_device'
        status['action'] = 'Connect Honor 100 Pro with USB debugging and authorize the computer.'
    (destination / 'acceptance.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
