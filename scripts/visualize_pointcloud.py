"""
Visualise ZipDepth depth as a coloured 3D point cloud (Open3D).

Inputs:
    image / video / folder of images   run the model, then back-project
    .npy / folder of .npy              reuse raw depth saved by infer.py --save-raw;
                                       --rgb gives the matching image, video or
                                       image folder for colours

ZipDepth outputs relative inverse depth: the cloud is placed in a virtual
[--near, --far] range, so its shape is meaningful but its size is not metric.

Usage:
    # Single image: interactive viewer + PLY file
    python scripts/visualize_pointcloud.py --input assets/examples/im0.jpg --save output/pointcloud/im0.ply

    # Video: animated cloud, also recorded to MP4 from an oblique viewpoint
    python scripts/visualize_pointcloud.py --input assets/examples/clip.mp4 --yaw 25 --pitch 10 \
        --record output/pointcloud/clip_cloud.mp4

    # Raw depth from `infer.py --save-raw` (colours taken from the source video)
    python scripts/visualize_pointcloud.py --input assets/examples/clip_depth_frames --rgb assets/examples/clip.mp4

    # Fine-tuned course checkpoint (NPU structure)
    python scripts/visualize_pointcloud.py --checkpoint output/experiments/temporal_0.1/best.pth --npu \
        --input assets/examples/clip.mp4 --smooth

Viewer: mouse drag rotates, wheel zooms, Space pauses/resumes a sequence, Q or Esc quits.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import itertools
import time

import cv2
import numpy as np

from zipdepth.utils.colormap import depth_to_colormap
from zipdepth.utils.pointcloud import (OPENCV_TO_OPENGL, InverseDepthMapper, depth_to_points,
                                       intrinsics_from_fov, scale_intrinsics)

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
DEFAULT_FPS = 30.0


# ── Frame sources: each yields (bgr or None, inverse depth or None) ───────────

def read_image(path):
    # imdecode instead of imread so non-ASCII Windows paths work.
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot load image: {path}')
    return image


def list_files(folder, extensions):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in extensions)


def iter_video(path):
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


def video_fps(path):
    if path is None or Path(path).suffix.lower() not in VIDEO_EXTS:
        return DEFAULT_FPS
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if np.isfinite(fps) and fps > 0 else DEFAULT_FPS


def iter_rgb(path):
    path = Path(path)
    if path.is_dir():
        return (read_image(p) for p in list_files(path, IMAGE_EXTS))
    if path.suffix.lower() in VIDEO_EXTS:
        return iter_video(path)
    return itertools.repeat(read_image(path))


def pair_with_rgb(depths, rgb_path):
    if rgb_path is None:
        return ((None, d) for d in depths)
    return zip(iter_rgb(rgb_path), depths)


def open_source(input_path, rgb_path):
    """Return (frames, is_sequence, fps)."""
    suffix = input_path.suffix.lower()
    if input_path.is_dir():
        depth_files = list_files(input_path, {'.npy'})
        if depth_files:
            depths = (np.load(p) for p in depth_files)
            return pair_with_rgb(depths, rgb_path), True, video_fps(rgb_path)
        images = list_files(input_path, IMAGE_EXTS)
        if not images:
            raise FileNotFoundError(f'No .npy depth or images found in {input_path}')
        return ((read_image(p), None) for p in images), True, DEFAULT_FPS
    if suffix == '.npy':
        return pair_with_rgb(iter([np.load(input_path)]), rgb_path), False, DEFAULT_FPS
    if suffix in VIDEO_EXTS:
        return ((f, None) for f in iter_video(input_path)), True, video_fps(input_path)
    if suffix in IMAGE_EXTS:
        return iter([(read_image(input_path), None)]), False, DEFAULT_FPS
    raise ValueError(f'Unsupported input: {input_path}')


def limit_size(image, max_side):
    h, w = image.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return image
    scale = max_side / max(h, w)
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


# ── Open3D viewer ─────────────────────────────────────────────────────────────

class CloudViewer:
    def __init__(self, image_size, point_size, visible, window_width=960):
        import open3d as o3d
        self.o3d = o3d
        h, w = image_size
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        # Window keeps the image aspect ratio so the initial view matches the camera.
        if not self.vis.create_window('ZipDepth point cloud', window_width,
                                      round(window_width * h / w), visible=visible):
            raise RuntimeError('Open3D could not create a window (no OpenGL context?)')
        option = self.vis.get_render_option()
        option.point_size = point_size
        option.background_color = np.zeros(3)
        self.cloud = o3d.geometry.PointCloud()
        self.added = False
        self.paused = False
        self.quit = False
        self.vis.register_key_callback(ord(' '), self._toggle_pause)
        self.vis.register_key_callback(ord('Q'), self._request_quit)
        self.vis.register_key_callback(256, self._request_quit)  # GLFW Esc

    def _toggle_pause(self, _):
        self.paused = not self.paused
        return False

    def _request_quit(self, _):
        self.quit = True
        return False

    def show(self, points, colors, camera):
        self.cloud.points = self.o3d.utility.Vector3dVector(points.astype(np.float64))
        self.cloud.colors = self.o3d.utility.Vector3dVector(colors.astype(np.float64))
        if self.added:
            self.vis.update_geometry(self.cloud)
        else:
            self.vis.add_geometry(self.cloud)
            self._set_camera(**camera)
            self.added = True
        return self.refresh()

    def _set_camera(self, intrinsics, image_width, pivot_depth, yaw, pitch, pullback, near, far):
        """Start at the original camera pose, orbited by yaw/pitch around the scene centre
        and moved ``pullback`` units backwards along the viewing direction."""
        control = self.vis.get_view_control()
        params = control.convert_to_pinhole_camera_parameters()
        win_w, win_h = params.intrinsic.width, params.intrinsic.height
        s = win_w / image_width
        fx, fy, _, _ = intrinsics
        params.intrinsic.set_intrinsics(win_w, win_h, fx * s, fy * s, win_w / 2 - 0.5, win_h / 2 - 0.5)

        a, b = np.deg2rad(yaw), np.deg2rad(pitch)
        rot_y = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        rot_x = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        rotation = rot_x @ rot_y
        pivot = np.array([0.0, 0.0, pivot_depth])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation @ np.diag(OPENCV_TO_OPENGL.astype(np.float64))
        extrinsic[:3, 3] = pivot - rotation @ pivot + np.array([0.0, 0.0, pullback])
        params.extrinsic = extrinsic

        if hasattr(control, 'set_constant_z_near'):
            control.set_constant_z_near(near * 0.05)
            control.set_constant_z_far(far * 10)
        if not control.convert_from_pinhole_camera_parameters(params, allow_arbitrary=True):
            print('  Warning: could not set the initial viewpoint; using Open3D default.')

    def refresh(self):
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive and not self.quit

    def wait_while_paused(self):
        while self.paused and self.refresh():
            time.sleep(0.01)
        return not self.quit

    def run_until_closed(self):
        while self.refresh():
            time.sleep(0.01)

    def capture(self):
        buffer = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))
        return cv2.cvtColor((np.clip(buffer, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def close(self):
        self.vis.destroy_window()


def save_cloud(path, points, colors):
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f'Failed to write {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='ZipDepth 3D point cloud visualisation')
    p.add_argument('--input', required=True,
                   help='Image, video, image folder, .npy depth or folder of .npy depth')
    p.add_argument('--rgb', default=None,
                   help='Colour source for .npy input: image, video or image folder '
                        '(depth colormap is used if omitted)')
    p.add_argument('--checkpoint', default='checkpoints/zipdepth_base.pth')
    p.add_argument('--npu', action='store_true',
                   help='Unfold-free upsampling (zipdepth_base_npu.pth and course fine-tuned weights)')
    p.add_argument('--device', default=None, choices=['cuda', 'cpu'],
                   help='Default: cuda if available, else cpu')
    p.add_argument('--input-size', type=int, default=384,
                   help='Shorter side of the model input (aspect ratio preserved)')
    p.add_argument('--max-side', type=int, default=640,
                   help='Downscale frames so the longer side is at most this before inference (0 = keep)')

    g = p.add_argument_group('geometry')
    g.add_argument('--fov', type=float, default=60.0,
                   help='Horizontal field of view in degrees, used when --fx is not given')
    g.add_argument('--fx', type=float, default=None,
                   help='Focal length in pixels of the original input (or of the .npy depth without --rgb)')
    g.add_argument('--fy', type=float, default=None, help='Defaults to --fx')
    g.add_argument('--cx', type=float, default=None, help='Defaults to the image centre')
    g.add_argument('--cy', type=float, default=None, help='Defaults to the image centre')
    g.add_argument('--near', type=float, default=1.0, help='Virtual depth of the nearest points')
    g.add_argument('--far', type=float, default=10.0, help='Virtual depth of the farthest points')
    g.add_argument('--range-ema', type=float, default=0.1,
                   help='Smoothing of the depth range across video frames (1 = per-frame range)')
    g.add_argument('--stride', type=int, default=2, help='Keep every N-th pixel per axis')
    g.add_argument('--edge-threshold', type=float, default=0.1,
                   help='Drop points at depth discontinuities (relative jump, 0 = keep all)')
    g.add_argument('--smooth', action='store_true',
                   help='Gated temporal EMA on video depth (same filter as the course demo)')

    v = p.add_argument_group('output')
    v.add_argument('--point-size', type=float, default=None,
                   help='Rendered point size in pixels (default: twice the on-screen point spacing)')
    v.add_argument('--window-width', type=int, default=960,
                   help='Viewer width in pixels; height follows the image aspect ratio')
    v.add_argument('--yaw', type=float, default=0.0,
                   help='Initial viewpoint orbit around the scene centre, degrees')
    v.add_argument('--pitch', type=float, default=0.0,
                   help='Initial viewpoint orbit around the scene centre, degrees')
    v.add_argument('--pullback', type=float, default=0.0,
                   help='Move the initial viewpoint backwards by this virtual depth (e.g. 2 with yaw/pitch)')
    v.add_argument('--save', default=None,
                   help='PLY/PCD file for one frame, or a directory for a sequence')
    v.add_argument('--record', default=None,
                   help='Capture the 3D view: .mp4 for a sequence, .png/.jpg for a single frame')
    v.add_argument('--no-display', action='store_true',
                   help='No interactive window (use with --save / --record)')
    v.add_argument('--max-frames', type=int, default=None)
    v.add_argument('--frame-step', type=int, default=1, help='Process every N-th frame')
    return p.parse_args()


def build_intrinsics(args, depth_hw, source_hw):
    h, w = depth_hw
    if args.fx is None:
        return intrinsics_from_fov(w, h, args.fov)
    sh, sw = source_hw
    base = (args.fx, args.fy or args.fx,
            (sw - 1) / 2 if args.cx is None else args.cx,
            (sh - 1) / 2 if args.cy is None else args.cy)
    return scale_intrinsics(base, w / sw, h / sh)


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f'Input not found: {args.input}')
    if args.rgb is not None and not Path(args.rgb).exists():
        raise FileNotFoundError(f'RGB source not found: {args.rgb}')
    if args.no_display and not (args.save or args.record):
        raise SystemExit('--no-display needs --save and/or --record, otherwise nothing is produced')
    try:
        import open3d  # noqa: F401
    except ImportError:
        raise SystemExit(f'Open3D is not installed for this interpreter ({sys.executable}).\n'
                         f'Install it with:  "{sys.executable}" -m pip install open3d')

    frames, is_sequence, fps = open_source(input_path, args.rgb)
    frames = itertools.islice(frames, 0, None, max(args.frame_step, 1))
    if args.max_frames:
        frames = itertools.islice(frames, args.max_frames)

    record_path = Path(args.record) if args.record else None
    if record_path is not None:
        expected = {'.mp4', '.avi'} if is_sequence else {'.png', '.jpg', '.jpeg'}
        if record_path.suffix.lower() not in expected:
            raise SystemExit(f'--record for this input must end with one of {sorted(expected)}')
        record_path.parent.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save) if args.save else None
    if save_path is not None and is_sequence:
        save_path.mkdir(parents=True, exist_ok=True)

    predictor = None
    mapper = InverseDepthMapper(args.near, args.far, ema=args.range_ema if is_sequence else None)
    smoother = None
    if args.smooth:
        from zipdepth.course.runtime import GatedEMA
        smoother = GatedEMA()

    viewer = writer = None
    model_ms, frame_ms, point_counts = [], [], []
    try:
        for index, (bgr, inverse) in enumerate(frames):
            start = time.perf_counter()
            source_hw = bgr.shape[:2] if bgr is not None else None
            if inverse is None:
                if predictor is None:
                    import torch
                    from zipdepth.inference.predictor import DepthInference
                    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
                    predictor = DepthInference(checkpoint_path=args.checkpoint, device=device,
                                               input_size=args.input_size,
                                               upsample_unfold=not args.npu)
                bgr = limit_size(bgr, args.max_side)
                t0 = time.perf_counter()
                inverse = predictor.infer_image(bgr)
                model_ms.append((time.perf_counter() - t0) * 1000)
            inverse = np.squeeze(np.asarray(inverse, np.float32))
            h, w = inverse.shape
            source_hw = source_hw or (h, w)

            rgb = None
            if bgr is not None:
                if bgr.shape[:2] != (h, w):
                    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if smoother is not None:
                if rgb is None:
                    raise SystemExit('--smooth needs RGB frames (pass --rgb for .npy input)')
                inverse = smoother.update(rgb.astype(np.float32) / 255.0, inverse)

            depth = mapper(inverse)
            if rgb is None:
                lo, hi = mapper.bounds
                rgb = cv2.cvtColor(depth_to_colormap(inverse, cmap='Spectral', vmin=lo, vmax=hi),
                                   cv2.COLOR_BGR2RGB)

            intrinsics = build_intrinsics(args, (h, w), source_hw)
            points, colors = depth_to_points(depth, intrinsics, rgb, stride=args.stride,
                                             edge_threshold=args.edge_threshold)
            points *= OPENCV_TO_OPENGL
            point_counts.append(len(points))

            if save_path is not None:
                save_cloud(save_path / f'frame_{index:06d}.ply' if is_sequence else save_path,
                           points, colors)

            if viewer is None and (not args.no_display or record_path is not None):
                point_size = args.point_size
                if point_size is None:
                    # Points narrower than their on-screen spacing leave gaps that shimmer
                    # from frame to frame; twice the spacing closes them.
                    point_size = float(np.clip(2 * args.stride * args.window_width / w, 1, 10))
                viewer = CloudViewer((h, w), point_size, visible=not args.no_display,
                                     window_width=args.window_width)
            if viewer is not None:
                # Orbit pivot at the disparity midpoint of [near, far], independent of scene content.
                camera = dict(intrinsics=intrinsics, image_width=w,
                              pivot_depth=float(np.sqrt(args.near * args.far)), yaw=args.yaw,
                              pitch=args.pitch, pullback=args.pullback,
                              near=args.near, far=args.far)
                if not viewer.show(points, colors, camera):
                    break
                if record_path is not None:
                    shot = viewer.capture()
                    if not is_sequence:
                        cv2.imencode(record_path.suffix, shot)[1].tofile(str(record_path))
                    else:
                        if writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            writer = cv2.VideoWriter(str(record_path), fourcc,
                                                     fps / max(args.frame_step, 1),
                                                     (shot.shape[1], shot.shape[0]))
                            if not writer.isOpened():
                                raise RuntimeError(f'Cannot open video writer: {record_path}')
                        writer.write(shot)
                if not viewer.wait_while_paused():
                    break
            frame_ms.append((time.perf_counter() - start) * 1000)
            if is_sequence and (index + 1) % 30 == 0:
                print(f'  {index + 1} frames, {1000 / np.mean(frame_ms[-30:]):.1f} FPS', flush=True)

        if not point_counts:
            raise ValueError('No frames were processed')
        print(f'\nFrames: {len(point_counts)}  |  points/frame: {int(np.mean(point_counts)):,}')
        if model_ms:
            print(f'  Model inference: {np.mean(model_ms):.1f} ms/frame')
        print(f'  End-to-end:      {np.mean(frame_ms):.1f} ms/frame '
              f'({1000 / np.mean(frame_ms):.1f} FPS, includes rendering)')
        if save_path is not None:
            print(f'  Point cloud(s):  {save_path}')
        if record_path is not None:
            print(f'  Recording:       {record_path}')
        if viewer is not None and not args.no_display and not viewer.quit:
            print('Close the window (or press Q) to exit.')
            viewer.run_until_closed()
    finally:
        if writer is not None:
            writer.release()
        if viewer is not None:
            viewer.close()


if __name__ == '__main__':
    main()
