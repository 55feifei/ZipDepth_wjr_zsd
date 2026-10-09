"""Check final ONNX files against their metadata and packaged APK bytes."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path


class ModelAssetError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_models(models_dir: Path, prefix: str) -> dict:
    models = {}
    checkpoints = {}
    for size in (256, 384):
        onnx_path = models_dir / f'{prefix}_{size}.onnx'
        meta_path = models_dir / f'{prefix}_{size}.json'
        if not onnx_path.is_file() or onnx_path.stat().st_size < 1024:
            raise ModelAssetError(f'Real ONNX model missing or too small: {onnx_path}')
        if not meta_path.is_file():
            raise ModelAssetError(f'Model metadata missing: {meta_path}')
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        expected = str(metadata.get('onnx_sha256', '')).lower()
        actual = sha256_file(onnx_path)
        if len(expected) != 64 or actual != expected:
            raise ModelAssetError(f'Hash mismatch for {onnx_path.name}: file {actual} metadata {expected}')
        checkpoint = str(metadata.get('checkpoint_sha256', '')).lower()
        if len(checkpoint) != 64:
            raise ModelAssetError(f'checkpoint_sha256 missing in {meta_path.name}')
        checkpoints[size] = checkpoint
        models[str(size)] = {
            'sha256': actual,
            'bytes': onnx_path.stat().st_size,
            'checkpoint_sha256': checkpoint,
            'metadata': str(meta_path),
        }
    if len(set(checkpoints.values())) != 1:
        raise ModelAssetError(f'256/384 checkpoint_sha256 differ: {checkpoints}')
    return {
        'prefix': prefix,
        'checkpoint_sha256': next(iter(set(checkpoints.values()))),
        'models': models,
    }


def asset_sources(root: Path, prefix: str) -> dict:
    models = root / 'output' / 'models'
    return {
        'assets/zipdepth_256.onnx': models / f'{prefix}_256.onnx',
        'assets/zipdepth_384.onnx': models / f'{prefix}_384.onnx',
        'assets/turbo_lut.json': root / 'tests' / 'fixtures' / 'turbo_lut.json',
    }


def verify_apk_assets(apk_path: Path, sources: dict) -> dict:
    if not apk_path.is_file():
        raise ModelAssetError(f'APK missing: {apk_path}')
    assets = {}
    with zipfile.ZipFile(apk_path) as archive:
        found = {info.filename: info for info in archive.infolist()}
        for name, source in sources.items():
            info = found.get(name)
            if info is None:
                raise ModelAssetError(f'APK missing asset {name}')
            stored = info.compress_type == zipfile.ZIP_STORED
            if name.endswith('.onnx') and not stored:
                raise ModelAssetError(f'ONNX asset must be stored uncompressed: {name}')
            payload = archive.read(name)
            digest = hashlib.sha256(payload).hexdigest()
            source_digest = sha256_file(Path(source))
            if digest != source_digest:
                raise ModelAssetError(f'APK asset {name} hash {digest} != source {source_digest}')
            assets[name] = {
                'sha256': digest,
                'bytes': len(payload),
                'stored': stored,
                'matches_source': True,
            }
    return {'apk_sha256': sha256_file(apk_path), 'assets': assets}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sources = sub.add_parser('check-sources')
    sources.add_argument('--root', type=Path, default=Path('.'))
    sources.add_argument('--prefix', required=True)
    packaged = sub.add_parser('check-apk')
    packaged.add_argument('--root', type=Path, default=Path('.'))
    packaged.add_argument('--prefix', required=True)
    packaged.add_argument('--apk', type=Path, required=True)
    packaged.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'check-sources':
            report = verify_source_models(args.root / 'output' / 'models', args.prefix)
        else:
            report = verify_apk_assets(args.apk, asset_sources(args.root, args.prefix))
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    except ModelAssetError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
