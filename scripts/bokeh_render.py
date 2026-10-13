"""
Bokeh (synthetic depth of field) rendering driven by ZipDepth depth.

Classical scattering renderer of BokehMe (CVPR 2022) on ZipDepth relative inverse
depth; see zipdepth/bokeh.py. The Android app runs the same algorithm with
--downsample 2.

Usage:
    # Image: bokeh image + 4-panel figure (RGB | disparity | defocus | bokeh)
    python scripts/bokeh_render.py --input assets/examples/im0.jpg --focus 0.68,0.35

    # Image: refocus animation, focal plane sweeping from near to far
    python scripts/bokeh_render.py --input assets/examples/im0.jpg --sweep 60

    # Video: side-by-side RGB | bokeh, phone variant for speed
    python scripts/bokeh_render.py --input assets/examples/clip.mp4 --focus 0.4,0.6 --smooth --downsample 2

Focus is given as x,y in [0, 1] of the image (default: centre).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import time

import cv2
import numpy as np

from zipdepth.bokeh import FocusTracker, defocus_map, render_bokeh
from zipdepth.utils.colormap import depth_to_colormap

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot load image: {path}')
    return image


def write_image(path, bgr):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, bgr)[1].tofile(str(path))


def iter_frames(path):
    """Yield BGR frames from a video file or a folder of images."""
    if path.is_dir():
        for file in sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            yield read_image(file)
        return
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def fit_size(image, max_side, multiple):
    """Resize so the longer side is at most max_side and both sides divide by `multiple`."""
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w)) if max_side else 1.0
    nh = max(multiple, int(h * scale) // multiple * multiple)
    nw = max(multiple, int(w * scale) // multiple * multiple)
    if (nh, nw) == (h, w):
        return image
    return cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)


def to_bgr(rgb):
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def label(image, text):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def mark_focus(image, point):
    out = image.copy()
    h, w = out.shape[:2]
    center = (int(round(point[0] * (w - 1))), int(round(point[1] * (h - 1))))
    cv2.circle(out, center, 9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(out, center, 9, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def panel(rgb_bgr, disparity, defocus, bokeh_bgr, point, focus):
    limit = max(float(np.abs(defocus).max()), 1e-6)
    tiles = [
        label(mark_focus(rgb_bgr, point), 'RGB + focus point'),
        label(depth_to_colormap(disparity, cmap='Spectral', vmin=0, vmax=1), 'ZipDepth disparity (red = near)'),
        label(depth_to_colormap(defocus, cmap='coolwarm', vmin=-limit, vmax=limit, invert=False),
              'defocus (red: in front, blue: behind)'),
        label(bokeh_bgr, f'bokeh (focus disparity {focus:.2f})'),
    ]
    return np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])


def open_writer(path, fps, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'Cannot open video writer: {path}')
    return writer


def parse_point(text):
    try:
        x, y = (float(v) for v in text.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError('focus must be "x,y" with values in [0, 1]')
    if not (0 <= x <= 1 and 0 <= y <= 1):
        raise argparse.ArgumentTypeError('focus must be "x,y" with values in [0, 1]')
    return x, y


def parse_args():
    p = argparse.ArgumentParser(description='ZipDepth bokeh rendering (BokehMe classical renderer)')
    p.add_argument('--input', required=True, help='Image, video or folder of images')
    p.add_argument('--output', default=None,
                   help='Output file (default: output/bokeh/<name>_bokeh.jpg or .mp4)')
    p.add_argument('--checkpoint', default='checkpoints/zipdepth_base.pth')
    p.add_argument('--npu', action='store_true',
                   help='Unfold-free upsampling (zipdepth_base_npu.pth and course fine-tuned weights)')
    p.add_argument('--device', default=None, choices=['cuda', 'cpu'])
    p.add_argument('--input-size', type=int, default=384, help='Shorter side of the model input')
    p.add_argument('--max-side', type=int, default=640, help='Working resolution (longer side)')

    b = p.add_argument_group('bokeh')
    b.add_argument('--focus', type=parse_point, default=(0.5, 0.5), help='Focus point x,y in [0, 1]')
    b.add_argument('--strength', type=float, default=0.03,
                   help='Blur radius for the full disparity range, fraction of the longer side')
    b.add_argument('--gamma', type=float, default=4.0, help='Highlight emphasis (BokehMe: 1-5)')
    b.add_argument('--downsample', type=int, default=1, choices=[1, 2, 4],
                   help='1 = full classical renderer, 2 = phone variant (about 4x faster)')
    b.add_argument('--smooth', action='store_true', help='Gated temporal EMA on video depth')
    b.add_argument('--sweep', type=int, default=0,
                   help='Image only: render an N-frame refocus animation (near to far and back)')
    b.add_argument('--no-panel', action='store_true', help='Image only: skip the 4-panel figure')

    v = p.add_argument_group('video')
    v.add_argument('--max-frames', type=int, default=None)
    v.add_argument('--with-depth', action='store_true', help='Add a disparity panel to the video')
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input not found: {args.input}')
    is_image = input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTS
    if not is_image and not input_path.is_dir() and input_path.suffix.lower() not in VIDEO_EXTS:
        raise SystemExit(f'Unsupported input: {input_path}')
    if args.sweep and not is_image:
        raise SystemExit('--sweep works on a single image')

    import torch
    from zipdepth.inference.predictor import DepthInference
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = DepthInference(checkpoint_path=args.checkpoint, device=device,
                               input_size=args.input_size, upsample_unfold=not args.npu,
                               warmup_iters=0 if is_image else 3)
    out_root = Path('output/bokeh')
    render = dict(strength=args.strength, gamma=args.gamma, downsample=args.downsample, device=device)

    if is_image:
        bgr = fit_size(read_image(input_path), args.max_side, args.downsample)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inverse = predictor.infer_image(bgr)
        disparity, focus = FocusTracker().update(inverse, args.focus)
        start = time.perf_counter()
        bokeh = render_bokeh(rgb, disparity, focus, **render)
        print(f'Bokeh render: {(time.perf_counter() - start) * 1000:.0f} ms at {bgr.shape[1]}x{bgr.shape[0]}')
        output = Path(args.output) if args.output else out_root / f'{input_path.stem}_bokeh.jpg'
        bokeh_bgr = to_bgr(bokeh)
        write_image(output, bokeh_bgr)
        print(f'Bokeh image: {output}')
        if not args.no_panel:
            figure = panel(bgr, disparity, defocus_map(disparity, focus, args.strength), bokeh_bgr,
                           args.focus, focus)
            figure_path = output.with_name(f'{output.stem}_panel.jpg')
            write_image(figure_path, figure)
            print(f'Panel:       {figure_path}')
        if args.sweep:
            sweep_path = output.with_name(f'{input_path.stem}_refocus.mp4')
            writer = open_writer(sweep_path, 24, (bgr.shape[1] * 2, bgr.shape[0]))
            try:
                for i in range(args.sweep):
                    phase = i / max(args.sweep - 1, 1)
                    target = abs(2 * phase - 1)  # disparity 1 -> 0 -> 1: near, far, near
                    frame = render_bokeh(rgb, disparity, target, **render)
                    writer.write(np.hstack([label(bgr, 'RGB'),
                                            label(to_bgr(frame), f'focus disparity {target:.2f}')]))
                    print(f'  refocus frame {i + 1}/{args.sweep}', end='\r', flush=True)
            finally:
                writer.release()
            print(f'\nRefocus animation: {sweep_path}')
        return

    fps = 30.0
    if input_path.is_file():
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
    output = Path(args.output) if args.output else out_root / f'{input_path.stem}_bokeh.mp4'
    tracker = FocusTracker()
    smoother = None
    if args.smooth:
        from zipdepth.course.runtime import GatedEMA
        smoother = GatedEMA()
    writer = None
    times = []
    try:
        for index, frame in enumerate(iter_frames(input_path)):
            if args.max_frames and index >= args.max_frames:
                break
            start = time.perf_counter()
            bgr = fit_size(frame, args.max_side, args.downsample)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            inverse = predictor.infer_image(bgr)
            if smoother is not None:
                inverse = smoother.update(rgb, inverse)
            snap = smoother is not None and smoother.scene_reset and index > 0
            disparity, focus = tracker.update(inverse, args.focus, snap=snap)
            bokeh = render_bokeh(rgb, disparity, focus, **render)
            tiles = [label(mark_focus(bgr, args.focus), 'RGB + focus point')]
            if args.with_depth:
                tiles.append(label(depth_to_colormap(disparity, cmap='Spectral', vmin=0, vmax=1),
                                   'ZipDepth disparity'))
            tiles.append(label(to_bgr(bokeh), f'bokeh (focus {focus:.2f})'))
            canvas = np.hstack(tiles)
            if writer is None:
                writer = open_writer(output, fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            times.append((time.perf_counter() - start) * 1000)
            if (index + 1) % 30 == 0:
                print(f'  {index + 1} frames, {np.mean(times[-30:]):.0f} ms/frame', flush=True)
    finally:
        if writer is not None:
            writer.release()
    if not times:
        raise ValueError('No frames were processed')
    print(f'\nFrames: {len(times)}  |  {np.mean(times):.0f} ms/frame (depth + bokeh)')
    print(f'Bokeh video: {output}')


if __name__ == '__main__':
    main()
