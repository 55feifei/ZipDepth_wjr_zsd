import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from zipdepth.course.model_assets import ModelAssetError, verify_apk_assets, verify_source_models

ROOT = Path(__file__).resolve().parents[1]
PLAN_384 = '1c8213a5b222f4b63ba95e1edb68c0d7dfa7e7f91c596c0d0de1bf7d96ae3013'


class ModelAssetTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)

    def _write_pair(self, directory, prefix, payload, metadata_hash=None, checkpoint='a' * 64):
        directory.mkdir(parents=True)
        digest = hashlib.sha256(payload).hexdigest()
        for size in (256, 384):
            onnx_path = directory / f'{prefix}_{size}.onnx'
            onnx_path.write_bytes(payload)
            meta = {'onnx_sha256': metadata_hash or digest, 'checkpoint_sha256': checkpoint}
            (directory / f'{prefix}_{size}.json').write_text(json.dumps(meta), encoding='utf-8')
        return digest

    def test_matching_metadata_passes(self):
        root = self.root / 'matching'
        digest = self._write_pair(root, 'zipdepth_final', b'onnx-bytes-' + b'0123456789' * 120)
        report = verify_source_models(root, 'zipdepth_final')
        self.assertEqual(report['models']['384']['sha256'], digest)
        self.assertEqual(report['checkpoint_sha256'], 'a' * 64)

    def test_metadata_mismatch_is_rejected(self):
        root = self.root / 'mismatch'
        self._write_pair(root, 'zipdepth_final', b'onnx-bytes-' + b'0123456789' * 120, metadata_hash='b' * 64)
        with self.assertRaisesRegex(ModelAssetError, 'Hash mismatch'):
            verify_source_models(root, 'zipdepth_final')

    def test_missing_model_is_rejected(self):
        root = self.root / 'missing'
        root.mkdir(parents=True)
        with self.assertRaises(ModelAssetError):
            verify_source_models(root, 'zipdepth_final')

    def test_checkpoint_mismatch_is_rejected(self):
        root = self.root / 'checkpoint'
        payload = b'onnx-bytes-' + b'0123456789' * 120
        digest = hashlib.sha256(payload).hexdigest()
        root.mkdir(parents=True)
        for size, checkpoint in ((256, 'a' * 64), (384, 'b' * 64)):
            (root / f'zipdepth_final_{size}.onnx').write_bytes(payload)
            (root / f'zipdepth_final_{size}.json').write_text(json.dumps({
                'onnx_sha256': digest, 'checkpoint_sha256': checkpoint}), encoding='utf-8')
        with self.assertRaisesRegex(ModelAssetError, 'checkpoint_sha256 differ'):
            verify_source_models(root, 'zipdepth_final')

    def test_apk_requires_stored_matching_bytes(self):
        folder = self.root / 'apk'
        folder.mkdir(parents=True)
        source = folder / 'model.onnx'
        source.write_bytes(b'model-bytes')
        lut = folder / 'turbo_lut.json'
        lut.write_bytes(b'{}')
        good = folder / 'good.apk'
        with zipfile.ZipFile(good, 'w', compression=zipfile.ZIP_STORED) as archive:
            archive.write(source, 'assets/zipdepth_256.onnx')
            archive.write(source, 'assets/zipdepth_384.onnx')
            archive.write(lut, 'assets/turbo_lut.json')
        report = verify_apk_assets(good, {
            'assets/zipdepth_256.onnx': source,
            'assets/zipdepth_384.onnx': source,
            'assets/turbo_lut.json': lut,
        })
        self.assertTrue(report['assets']['assets/zipdepth_384.onnx']['matches_source'])
        self.assertTrue(report['assets']['assets/zipdepth_384.onnx']['stored'])
        compressed = folder / 'compressed.apk'
        with zipfile.ZipFile(compressed, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(source, 'assets/zipdepth_256.onnx')
            archive.write(source, 'assets/zipdepth_384.onnx')
            archive.write(lut, 'assets/turbo_lut.json')
        with self.assertRaises(ModelAssetError):
            verify_apk_assets(compressed, {
                'assets/zipdepth_256.onnx': source,
                'assets/zipdepth_384.onnx': source,
                'assets/turbo_lut.json': lut,
            })

    def test_real_final_384_matches_plan_hash(self):
        report = verify_source_models(ROOT / 'output' / 'models', 'zipdepth_final')
        self.assertEqual(report['models']['384']['sha256'], PLAN_384)
        self.assertEqual(report['models']['256']['checkpoint_sha256'], report['models']['384']['checkpoint_sha256'])


if __name__ == '__main__':
    unittest.main()
