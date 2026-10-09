"""Summarize real phone measurements without collapsing distinct test configurations."""
import math

TARGET_FPS = 15
REQUIRED_FINAL = (
    ('384_smoothing_on', 384, True),
    ('384_smoothing_off', 384, False),
    ('256_smoothing_on', 256, True),
    ('256_smoothing_off', 256, False),
)
MANUAL_CHECKS = (
    'first_launch_384',
    'camera_granted',
    'camera_denied_retry',
    'camera_permanent_deny_settings',
    'switch_384_256',
    'smoothing_toggle',
    'rotation',
    'background_foreground',
    'benchmark_background_incomplete',
    'offline_inference',
    'export_measurements',
    'five_minute_stable',
)
STAGE_COLUMNS = ('preprocess_ms', 'inference_ms', 'postprocess_ms', 'ui_queue_ms', 'pipeline_ms')


def _percentile(values, fraction):
    index = (len(values) - 1) * fraction
    low = int(index)
    return values[low] + (index - low) * (values[min(low + 1, len(values) - 1)] - values[low])


def assess_measurement(summary, rows, model_sha_matches):
    reasons = []
    if not summary.get('completed'):
        reasons.append('phone_test_interrupted')
    try:
        span = float(summary.get('observed_update_span_s', 0))
    except (TypeError, ValueError):
        span = -1
    if not math.isfinite(span) or span < 299:
        reasons.append('less_than_five_minutes')
    if rows is None:
        reasons.append('missing_frames_csv')
        rows = []
    updates = summary.get('depth_updates')
    if not rows or len(rows) != updates:
        reasons.append('frame_count_mismatch')
    if not model_sha_matches:
        reasons.append('model_provenance_mismatch')
    if not summary.get('backend'):
        reasons.append('backend_missing')
    smoothing = summary.get('smoothing')
    if not isinstance(smoothing, bool):
        reasons.append('smoothing_state_missing')
    fps = summary.get('display_update_fps')
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) < 0:
        reasons.append('invalid_fps')
        fps_value = None
    else:
        fps_value = float(fps)
    stages = {}
    for column in STAGE_COLUMNS:
        try:
            values = sorted(float(row[column]) for row in rows)
        except (KeyError, TypeError, ValueError):
            reasons.append(f'invalid_{column}')
            continue
        if not values or not all(math.isfinite(value) and value >= 0 for value in values):
            reasons.append(f'invalid_{column}')
            continue
        stages[column] = {'mean': sum(values) / len(values), 'p50': _percentile(values, .5), 'p95': _percentile(values, .95)}
    try:
        size = int(summary['resolution'])
    except (KeyError, TypeError, ValueError):
        size = None
        reasons.append('resolution_missing')
    flavor = str(summary.get('model_flavor', 'zipdepth_baseline'))
    if flavor.startswith('zipdepth_'):
        flavor = flavor[len('zipdepth_'):]
    return {
        'size': size,
        'model_flavor': flavor,
        'smoothing': smoothing if isinstance(smoothing, bool) else None,
        'model_sha256': summary.get('model_sha256'),
        'valid_continuous_measurement': not reasons,
        'invalid_reasons': reasons,
        'fps': fps_value,
        'stages_ms': stages,
        'p50_ms': summary.get('pipeline_p50_ms'),
        'p95_ms': summary.get('pipeline_p95_ms'),
        'backend': summary.get('backend'),
        'backend_at_start': summary.get('backend_at_start'),
        'fallback_reason': summary.get('fallback_reason'),
        'battery_temp_start_c': summary.get('battery_temp_start_c'),
        'battery_temp_end_c': summary.get('battery_temp_end_c'),
        'completed': bool(summary.get('completed')),
        'observed_update_span_s': span,
        'depth_updates': updates,
    }


def latest_by_config(measurements):
    latest = {}
    for measure in sorted(measurements, key=lambda item: item.get('summary', '')):
        latest[(measure.get('model_flavor'), measure.get('size'), measure.get('smoothing'))] = measure
    return latest


def _meets_target(measure):
    if not measure or not measure.get('valid_continuous_measurement'):
        return False
    fps = measure.get('fps')
    return isinstance(fps, (int, float)) and math.isfinite(fps) and fps >= TARGET_FPS


def manual_is_complete(document):
    if not isinstance(document, dict) or document.get('complete') is not True:
        return False
    checks = document.get('checks')
    if not isinstance(checks, dict):
        return False
    return all(checks.get(name) is True for name in MANUAL_CHECKS)


def build_acceptance(measurements, manual_complete, demo_present, device=None, device_connected=False):
    latest = latest_by_config(measurements)
    configurations = {}
    met_sizes = []
    for name, size, smoothing in REQUIRED_FINAL:
        measure = latest.get(('final', size, smoothing))
        met = _meets_target(measure)
        if met:
            met_sizes.append(size)
        configurations[name] = {
            'size': size,
            'smoothing': smoothing,
            'present': measure is not None,
            'valid_continuous_measurement': bool(measure and measure.get('valid_continuous_measurement')),
            'met_15_fps': met,
            'fps': None if measure is None else measure.get('fps'),
            'summary': None if measure is None else measure.get('summary'),
            'invalid_reasons': [] if measure is None else measure.get('invalid_reasons', []),
            'stages_ms': None if measure is None else measure.get('stages_ms'),
            'backend': None if measure is None else measure.get('backend'),
            'backend_at_start': None if measure is None else measure.get('backend_at_start'),
            'fallback_reason': None if measure is None else measure.get('fallback_reason'),
            'battery_temp_start_c': None if measure is None else measure.get('battery_temp_start_c'),
            'battery_temp_end_c': None if measure is None else measure.get('battery_temp_end_c'),
            'model_sha256': None if measure is None else measure.get('model_sha256'),
            'observed_update_span_s': None if measure is None else measure.get('observed_update_span_s'),
        }
    automatic = all(item['valid_continuous_measurement'] for item in configurations.values())
    overall = bool(automatic and manual_complete and demo_present)
    remaining = []
    if not device_connected:
        remaining.append('连接荣耀 100 Pro，开启 USB 调试并在手机上授权这台电脑。')
    if not automatic:
        remaining.append('完成四组有效五分钟测量：384/256 与时序平滑开/关，且各组原始 CSV、JSON、profiling 文件都保留。')
    if not manual_complete:
        remaining.append('逐项完成人工功能验收，并把通过结果写入 output/device/manual_acceptance.json。')
    if not demo_present:
        remaining.append('在测速之后录制并拉取 120 秒演示视频 output/device/demo.mp4。')
    return {
        'phone_verified': overall,
        'overall_complete': overall,
        'device_connected': bool(device_connected),
        'device': device,
        'measurements': measurements,
        'configurations': configurations,
        'automatic_measurement_complete': automatic,
        'manual_functional_acceptance_complete': bool(manual_complete),
        'demo_recording_present': bool(demo_present),
        'size_384_meets_15_fps': configurations['384_smoothing_on']['met_15_fps'],
        'size_256_meets_15_fps': configurations['256_smoothing_on']['met_15_fps'],
        'recommended_final_size': max(met_sizes) if met_sizes else None,
        'recommendation_policy': '只在实测达到 15 FPS 的最终模型配置中取最大输入边长；没有任何配置达到时为 null，不把 256 当作达标推荐。15 FPS 是项目性能目标，不是课程硬性门槛。',
        'raw_measurements_preserved': True,
        'remaining': remaining,
    }
