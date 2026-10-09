"""Small-memory, reproducible Windows training for paired ZipDepth frames."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from zipdepth.model.architecture import create_model
from zipdepth.utils.model_utils import strip_state_dict_prefixes
from .losses import TemporalDepthLoss


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _manifest_sha256(manifest):
    if not manifest:
        return None
    return hashlib.sha256(Path(manifest).read_bytes()).hexdigest()


def freeze_bn_statistics(model):
    """Keep pretrained running statistics; BN weight/bias remain trainable."""
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            layer.eval()


def capture_rng():
    state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (state[0], torch.from_numpy(state[1].astype(np.int64)), state[2], state[3], state[4]),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], ns[1].cpu().numpy().astype(np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def strict_load_mobile_model(checkpoint: str, device: str = 'cpu'):
    model = create_model(variant='base', global_mode='balanced', upsample_unfold=False)
    content = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = strip_state_dict_prefixes(content.get('model_state_dict', content))
    model.load_state_dict(state, strict=True)
    return model.to(device)


def enable_deterministic_mobile_pooling(model):
    """Use the equivalent fixed average pool for the exact 2x cross-scale map.

    CUDA adaptive_avg_pool2d backward is nondeterministic. Course input sizes
    are multiples of 32, so this pool always maps divisible grids at ratio 2.
    Model parameters and inference mathematics remain unchanged.
    """
    cross_scale = getattr(getattr(model, 'encoder', None), 'cross_scale', None)
    if cross_scale is None:
        return

    def forward(layer, x_high, x_low):
        ratio_h = x_high.shape[-2] // x_low.shape[-2]
        ratio_w = x_high.shape[-1] // x_low.shape[-1]
        if x_high.shape[-2] != x_low.shape[-2] * ratio_h or x_high.shape[-1] != x_low.shape[-1] * ratio_w:
            raise ValueError('Deterministic mobile training requires input size divisible by 32')
        low_up = F.interpolate(layer.low_to_high(x_low), size=x_high.shape[-2:], mode='nearest')
        high_down = F.avg_pool2d(layer.high_to_low(x_high), (ratio_h, ratio_w), (ratio_h, ratio_w))
        return x_high + low_up * 0.3, x_low + high_down * 0.3

    cross_scale.forward = types.MethodType(forward, cross_scale)


def aligned_absrel(pred, target, mask, min_depth=0.1, max_depth=80.0):
    """Per-frame affine alignment in inverse-depth space, then metric depth."""
    values = []
    for p, t, m in zip(pred.flatten(0, 1), target.flatten(0, 1), mask.flatten(0, 1)):
        good = m.bool() & torch.isfinite(p) & torch.isfinite(t) & (t > 0)
        pv, tv = p[good].double(), t[good].double()
        if pv.numel() < 10:
            continue
        pc, tc = pv - pv.mean(), tv - tv.mean()
        scale = (pc * tc).sum() / pc.square().sum().clamp_min(1e-12)
        shift = tv.mean() - scale * pv.mean()
        depth_p = (scale * pv + shift).clamp_min(1.0 / max_depth).reciprocal().clamp(min_depth, max_depth)
        depth_t = tv.reciprocal().clamp(min_depth, max_depth)
        values.append(((depth_p - depth_t).abs() / depth_t).mean())
    return torch.stack(values).mean().item() if values else float('inf')


def _loader(dataset, indices=None, seed=42):
    # A separate generator avoids consuming model RNG when a loader iterator is
    # recreated during mid-epoch resume. num_workers=0 preserves exact augmentation RNG.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=1, sampler=indices, shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator)


def _device_batch(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def _predict(model, batch, amp, device):
    image = batch['image']
    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        prediction = model(image.flatten(0, 1))
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Model produced non-finite predictions')
    return prediction.float().reshape(image.shape[0], 2, 1, *image.shape[-2:])


@torch.no_grad()
def validate(model, dataset, criterion, device, amp=False, max_pairs=64):
    model.eval()
    if len(dataset) == 0:
        raise ValueError('Validation split is empty; split by complete trajectories first')
    if max_pairs and len(dataset) > max_pairs:
        indices = np.linspace(0, len(dataset) - 1, max_pairs, dtype=int).tolist()
    else:
        indices = list(range(len(dataset)))
    sums = {'spatial': 0.0, 'temporal': 0.0, 'total': 0.0, 'absrel': 0.0}
    pairs, skipped_pairs, temporal_pixels = 0, 0, 0
    for batch in _loader(dataset, indices):
        batch = _device_batch(batch, device)
        usable = batch['mask'].bool() & torch.isfinite(batch['target']) & (batch['target'] > 0)
        if not (usable.flatten(3).sum(3) >= 10).any():
            skipped_pairs += 1
            continue
        prediction = _predict(model, batch, amp, device)
        _, metrics = criterion(prediction, batch['target'], batch['mask'], batch['flow'], batch['flow_mask'])
        for name in ('spatial', 'temporal', 'total'):
            sums[name] += metrics[name].item()
        sums['absrel'] += aligned_absrel(prediction, batch['target'], batch['mask'])
        temporal_pixels += int(metrics['valid_temporal_pixels'])
        pairs += 1
    if pairs == 0:
        raise ValueError('Validation contains no usable depth targets (at least 10 valid pixels per frame required)')
    if temporal_pixels == 0:
        raise ValueError('Validation contains no valid flow-aligned pixels; cannot measure temporal stability')
    return {**{key: val / pairs for key, val in sums.items()}, 'pairs': pairs,
            'skipped_pairs': skipped_pairs, 'valid_temporal_pixels': temporal_pixels,
            'temporal_alignment': 'shared_per_pair_median_mad', 'spatial_alignment': 'per_frame_affine_inverse_depth'}


def measure_subset_validation(checkpoint, config):
    """Score a checkpoint with the same fixed validation subset used in training.

    The returned AbsRel is the only scale that may gate in-training ``best.pth``.
    Full-split sequence metrics stay in the later selection pass.
    """
    seed = int(config.get('seed', 42))
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    amp = bool(config.get('amp', True)) and device.type == 'cuda'
    from .data import TartanAirPairDataset
    val_dataset = TartanAirPairDataset(
        manifest_path=config['manifest'], split='val', augment=False,
        input_size=int(config.get('input_size', 384)), seed=seed,
        max_depth=float(config.get('max_depth', 80.0)))
    model = strict_load_mobile_model(checkpoint, str(device))
    enable_deterministic_mobile_pooling(model)
    criterion = TemporalDepthLoss(temporal_weight=0.0)
    metrics = validate(model, val_dataset, criterion, device, amp, int(config.get('max_val_pairs', 64)))
    metrics['checkpoint'] = str(checkpoint)
    metrics['baseline_scale'] = 'same_subset_aligned_absrel'
    return metrics


def train_temporal(config: dict, train_dataset=None, val_dataset=None, model=None):
    """Run finite-budget training; injectable datasets/model enable smoke tests.

    Checkpoints are written only at optimizer-update boundaries. They store the
    exact next microbatch, permutation seed, optimizer, and all RNG states.
    """
    config = dict(config)
    # Paths alone cannot identify the dataset: download/index jobs can rewrite
    # the same manifest while independent experiments or resumes are running.
    config['manifest_sha256'] = _manifest_sha256(config.get('manifest'))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed = int(config.get('seed', 42))
    seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(bool(config.get('deterministic', True)))
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; check the ML1 environment and driver')
    amp = bool(config.get('amp', True)) and device.type == 'cuda'
    if amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 requested but unsupported on this CUDA device')
    output = Path(config['output']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'effective_config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    if train_dataset is None or val_dataset is None:
        from .data import TartanAirPairDataset
        kwargs = dict(manifest_path=config['manifest'], input_size=int(config.get('input_size', 384)),
                      seed=seed, max_depth=float(config.get('max_depth', 80.0)))
        train_dataset = TartanAirPairDataset(split='train', augment=True, **kwargs)
        val_dataset = TartanAirPairDataset(split='val', augment=False, **kwargs)
        if _manifest_sha256(config['manifest']) != config['manifest_sha256']:
            raise ValueError('Manifest changed while loading the datasets; freeze it before training')
    if not len(train_dataset):
        raise ValueError('Training split is empty')
    if not len(val_dataset):
        raise ValueError('Validation split is empty; split by complete trajectories first')
    if model is None:
        model = strict_load_mobile_model(config['checkpoint'], str(device))
    else:
        model = model.to(device)
    enable_deterministic_mobile_pooling(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get('lr', 1e-5)),
                                 weight_decay=float(config.get('weight_decay', 0.01)))
    criterion = TemporalDepthLoss(temporal_weight=float(config.get('temporal_weight', 0.1)),
                                  alpha_ssi=float(config.get('alpha_ssi', 1.0)),
                                  alpha_grad=float(config.get('alpha_grad', 2.0)),
                                  grad_scales=int(config.get('grad_scales', 4)))
    accumulation = int(config.get('grad_accum', 4))
    max_updates = int(config.get('max_updates', 2000))
    epochs = int(config.get('epochs', 5))
    if min(accumulation, max_updates, epochs) < 1:
        raise ValueError('grad_accum, max_updates, epochs must be positive')
    validate_every = int(config.get('validate_every', 100))
    val_pairs = int(config.get('max_val_pairs', 64))
    max_seconds = float(config.get('max_seconds', 0))
    start_epoch, cursor, updates, elapsed_before = 0, 0, 0, 0.0
    best_score, best_validation = float('inf'), None
    last_validation = None
    last_validated_update = -1
    resume_best_status = 'fresh_run'
    resume = config.get('resume')
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        previous = saved['config']
        immutable = ('seed', 'input_size', 'grad_accum', 'temporal_weight', 'lr', 'weight_decay', 'manifest', 'manifest_sha256')
        for name in immutable:
            if previous.get(name) != config.get(name):
                raise ValueError(f'Resume changes reproducibility-critical configuration: {name}')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        start_epoch, cursor, updates = saved['next_epoch'], saved['next_microbatch'], saved['global_step']
        elapsed_before = float(saved.get('elapsed_seconds', 0))
        best_score, best_validation = saved.get('best_score', float('inf')), saved.get('best_validation')
        last_validation = saved.get('validation')
        last_validated_update = updates if last_validation else -1
        destination_best = output / 'best.pth'
        source_best = Path(resume).resolve().parent / 'best.pth'
        if best_validation is not None:
            if source_best.is_file():
                if source_best != destination_best:
                    shutil.copy2(source_best, destination_best)
                    resume_best_status = 'copied_source_best'
                else:
                    resume_best_status = 'retained_source_best'
            elif saved.get('validation') == best_validation:
                torch.save(saved, destination_best)
                resume_best_status = 'resume_checkpoint_is_best'
            else:
                # The last model is not the historical best. Never label it as
                # best merely because its checkpoint contains a best score.
                best_score, best_validation = float('inf'), None
                resume_best_status = 'historical_best_unavailable_reset_selection'
                print('Historical best weights unavailable; selecting a new best from resumed validation.', flush=True)
        restore_rng(saved['rng_state'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    log_path = output / 'train.jsonl'
    baseline_absrel = config.get('baseline_absrel')
    checkpoint_every = int(config.get('checkpoint_every', 0))
    recorded_subset_updates = set()

    def elapsed():
        return elapsed_before + time.monotonic() - started

    def checkpoint(next_epoch, next_cursor, validation):
        nonlocal best_score, best_validation
        score = validation['temporal'] if baseline_absrel is not None else validation['spatial']
        eligible = baseline_absrel is None or validation['absrel'] <= float(baseline_absrel) * 1.02
        improved = eligible and math.isfinite(score) and score < best_score
        if improved:
            best_score, best_validation = score, validation
        saved = {'format_version': 1, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(), 'rng_state': capture_rng(),
                 'next_epoch': next_epoch, 'next_microbatch': next_cursor, 'global_step': updates,
                 'elapsed_seconds': elapsed(), 'config': config, 'validation': validation,
                 'best_score': best_score, 'best_validation': best_validation,
                 'model_spec': {'variant': 'base', 'global_mode': 'balanced', 'upsample_unfold': False}}
        destination = output / 'last.pth'
        temporary = output / 'last.pth.tmp'
        torch.save(saved, temporary)
        temporary.replace(destination)
        if improved:
            torch.save(saved, output / 'best.pth')
        snapshot = None
        if checkpoint_every > 0 and updates > 0 and updates % checkpoint_every == 0:
            folder = output / 'checkpoints'
            folder.mkdir(exist_ok=True)
            snapshot = f'checkpoints/update_{updates:05d}.pth'
            shutil.copy2(destination, output / snapshot)
            meta = {**validation, 'update': updates, 'checkpoint': snapshot, 'eligible': eligible,
                    'baseline_absrel': None if baseline_absrel is None else float(baseline_absrel),
                    'baseline_scale': 'same_subset_aligned_absrel' if baseline_absrel is not None else 'spatial_only',
                    'selection_score': score, 'saved_best': improved}
            (folder / f'update_{updates:05d}.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        if updates not in recorded_subset_updates:
            recorded_subset_updates.add(updates)
            record = {**validation, 'update': updates, 'eligible': eligible,
                      'baseline_absrel': None if baseline_absrel is None else float(baseline_absrel),
                      'baseline_scale': 'same_subset_aligned_absrel' if baseline_absrel is not None else 'spatial_only',
                      'selection_score': score, 'saved_best': improved, 'snapshot': snapshot}
            with (output / 'subset_validation.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record) + '\n')

    next_epoch, next_cursor = start_epoch, cursor
    stop = updates >= max_updates or (max_seconds > 0 and elapsed() >= max_seconds)
    for epoch in range(start_epoch, epochs):
        if stop:
            break
        if hasattr(train_dataset, 'set_epoch'):
            train_dataset.set_epoch(epoch)
        indices = list(range(len(train_dataset)))
        random.Random(seed + epoch).shuffle(indices)
        epoch_cursor = cursor if epoch == start_epoch else 0
        indices = indices[epoch_cursor:]
        loader = _loader(train_dataset, indices, seed + epoch)
        model.train()
        freeze_bn_statistics(model)
        optimizer.zero_grad(set_to_none=True)
        count = 0
        loss_sums = {}
        for microbatch, batch in enumerate(loader, start=epoch_cursor):
            batch = _device_batch(batch, device)
            prediction = _predict(model, batch, amp, device)
            loss, metrics = criterion(prediction, batch['target'], batch['mask'], batch['flow'], batch['flow_mask'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch={epoch}, microbatch={microbatch}')
            (loss / accumulation).backward()
            count += 1
            for name in ('spatial', 'temporal', 'total'):
                loss_sums[name] = loss_sums.get(name, 0.0) + metrics[name].item()
            is_epoch_end = microbatch + 1 == len(train_dataset)
            if count < accumulation and not is_epoch_end:
                continue
            if count < accumulation:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(accumulation / count)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            next_epoch, next_cursor = (epoch + 1, 0) if is_epoch_end else (epoch, microbatch + 1)
            record = {'update': updates, 'epoch': epoch, 'microbatch': microbatch,
                      **{name: value / count for name, value in loss_sums.items()},
                      'grad_norm': float(norm), 'elapsed_seconds': elapsed()}
            with log_path.open('a', encoding='utf-8') as log:
                log.write(json.dumps(record) + '\n')
            print(f"update {updates}/{max_updates}: spatial={record['spatial']:.5f}, "
                  f"temporal={record['temporal']:.5f}, elapsed={elapsed():.1f}s", flush=True)
            count, loss_sums = 0, {}
            stop = updates >= max_updates or (max_seconds > 0 and elapsed() >= max_seconds)
            if (validate_every > 0 and updates % validate_every == 0) or is_epoch_end or stop:
                last_validation = validate(model, val_dataset, criterion, device, amp, val_pairs)
                last_validated_update = updates
                checkpoint(next_epoch, next_cursor, last_validation)
                print(f'validation: {json.dumps(last_validation)}', flush=True)
                model.train()
                freeze_bn_statistics(model)
            if stop:
                break
        cursor = 0
    if last_validation is None or last_validated_update != updates:
        last_validation = validate(model, val_dataset, criterion, device, amp, val_pairs)
    # Persist even a no-op resume into a new output directory. Final optimizer
    # state and validation always refer to the last completed update.
    checkpoint(next_epoch, next_cursor, last_validation)
    summary = {'updates': updates, 'elapsed_seconds': elapsed(), 'validation': last_validation,
               'best_validation': best_validation, 'best_selection': 'temporal_with_absrel_gate' if baseline_absrel is not None else 'spatial',
               'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == 'cuda' else 0,
               'device': str(device), 'amp_dtype': 'bfloat16' if amp else 'float32',
               'resume_best_status': resume_best_status,
               'last_checkpoint': str(output / 'last.pth'), 'best_checkpoint': str(output / 'best.pth') if best_validation else None}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary
