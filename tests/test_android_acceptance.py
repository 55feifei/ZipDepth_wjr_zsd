import unittest
from zipdepth.course.android_acceptance import MANUAL_CHECKS, assess_measurement, build_acceptance, manual_is_complete


def measurement(path, size, smoothing, fps, valid=True, updates=2):
    summary = {
        'completed': valid,
        'observed_update_span_s': 299.4 if valid else 10,
        'depth_updates': updates,
        'resolution': size,
        'smoothing': smoothing,
        'display_update_fps': fps,
        'model_flavor': 'zipdepth_final',
        'model_sha256': 'abc',
        'backend': 'XNNPACK + CPU',
        'fallback_reason': None,
        'battery_temp_start_c': 31.0,
        'battery_temp_end_c': 35.5,
        'pipeline_p50_ms': 40,
        'pipeline_p95_ms': 70,
    }
    rows = [{
        'preprocess_ms': '1.0', 'inference_ms': '20.0', 'postprocess_ms': '2.0',
        'ui_queue_ms': '0.5', 'pipeline_ms': '23.5',
    } for _ in range(updates)]
    assessed = assess_measurement(summary, rows, model_sha_matches=True)
    assessed['summary'] = path
    return assessed


class AcceptanceSummaryTests(unittest.TestCase):
    def test_size_and_smoothing_stay_distinct(self):
        rows = [
            measurement('files/measurements/1/summary.json', 384, True, 12),
            measurement('files/measurements/2/summary.json', 384, False, 13),
            measurement('files/measurements/3/summary.json', 256, True, 16),
            measurement('files/measurements/4/summary.json', 256, False, 18),
        ]
        status = build_acceptance(rows, manual_complete=False, demo_present=False)
        self.assertEqual(status['configurations']['384_smoothing_on']['fps'], 12)
        self.assertEqual(status['configurations']['384_smoothing_off']['fps'], 13)
        self.assertEqual(status['configurations']['256_smoothing_on']['fps'], 16)
        self.assertEqual(status['configurations']['256_smoothing_off']['fps'], 18)
        self.assertTrue(status['automatic_measurement_complete'])
        self.assertFalse(status['size_384_meets_15_fps'])
        self.assertTrue(status['size_256_meets_15_fps'])
        self.assertEqual(status['recommended_final_size'], 256)

    def test_newer_run_replaces_only_the_same_configuration(self):
        rows = [
            measurement('files/measurements/1/summary.json', 384, True, 10),
            measurement('files/measurements/9/summary.json', 384, True, 17),
            measurement('files/measurements/2/summary.json', 384, False, 11),
        ]
        status = build_acceptance(rows, False, False)
        self.assertEqual(status['configurations']['384_smoothing_on']['fps'], 17)
        self.assertEqual(status['configurations']['384_smoothing_off']['fps'], 11)
        self.assertFalse(status['configurations']['256_smoothing_on']['present'])

    def test_no_config_at_target_does_not_recommend_256(self):
        rows = [
            measurement(f'files/measurements/{index}/summary.json', size, smoothing, fps)
            for index, (size, smoothing, fps) in enumerate([
                (384, True, 9), (384, False, 10), (256, True, 11), (256, False, 12)], start=1)
        ]
        status = build_acceptance(rows, False, False)
        self.assertIsNone(status['recommended_final_size'])
        self.assertFalse(status['size_384_meets_15_fps'])
        self.assertFalse(status['size_256_meets_15_fps'])
        self.assertNotEqual(status['recommended_final_size'], 256)

    def test_invalid_replacement_is_not_treated_as_passing(self):
        rows = [
            measurement('files/measurements/1/summary.json', 384, True, 20),
            measurement('files/measurements/2/summary.json', 384, True, 20, valid=False),
        ]
        status = build_acceptance(rows, False, False)
        self.assertFalse(status['configurations']['384_smoothing_on']['valid_continuous_measurement'])
        self.assertFalse(status['configurations']['384_smoothing_on']['met_15_fps'])

    def test_overall_requires_measurements_manual_checks_and_video(self):
        rows = [
            measurement(f'files/measurements/{index}/summary.json', size, smoothing, 16)
            for index, (size, smoothing) in enumerate([
                (384, True), (384, False), (256, True), (256, False)], start=1)
        ]
        incomplete = build_acceptance(rows, manual_complete=False, demo_present=False)
        self.assertTrue(incomplete['automatic_measurement_complete'])
        self.assertFalse(incomplete['manual_functional_acceptance_complete'])
        self.assertFalse(incomplete['demo_recording_present'])
        self.assertFalse(incomplete['overall_complete'])
        self.assertFalse(incomplete['phone_verified'])
        complete = build_acceptance(rows, manual_complete=True, demo_present=True, device_connected=True)
        self.assertTrue(complete['overall_complete'])
        self.assertTrue(complete['phone_verified'])
        self.assertEqual(complete['remaining'], [])

    def test_manual_file_must_pass_every_check(self):
        self.assertFalse(manual_is_complete(None))
        self.assertFalse(manual_is_complete({'complete': True, 'checks': {MANUAL_CHECKS[0]: True}}))
        document = {'complete': True, 'checks': {name: True for name in MANUAL_CHECKS}}
        self.assertTrue(manual_is_complete(document))
        document['checks']['rotation'] = False
        self.assertFalse(manual_is_complete(document))

    def test_missing_csv_is_invalid_without_hiding_other_configs(self):
        broken = assess_measurement({
            'completed': True, 'observed_update_span_s': 300, 'depth_updates': 2,
            'resolution': 384, 'smoothing': True, 'display_update_fps': 16,
            'model_flavor': 'zipdepth_final', 'model_sha256': 'abc', 'backend': 'CPU',
        }, None, True)
        broken['summary'] = 'files/measurements/1/summary.json'
        other = measurement('files/measurements/2/summary.json', 256, False, 16)
        status = build_acceptance([broken, other], False, False)
        self.assertIn('missing_frames_csv', status['configurations']['384_smoothing_on']['invalid_reasons'])
        self.assertEqual(status['configurations']['256_smoothing_off']['fps'], 16)


if __name__ == '__main__':
    unittest.main()
