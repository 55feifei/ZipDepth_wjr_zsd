"""
Open saved point clouds (.ply / .pcd) in an Open3D window.

Usage:
    # One file
    python scripts/view_pointcloud.py output/pointcloud/im0.ply

    # Folder of per-frame clouds from visualize_pointcloud.py --save <dir>: played as a sequence
    python scripts/view_pointcloud.py output/pointcloud/clip_ply --fps 10

Viewer: mouse drag rotates, wheel zooms, Space pauses/resumes a sequence, Q or Esc quits.
"""

import argparse
import time
from pathlib import Path

import numpy as np

CLOUD_EXTS = {'.ply', '.pcd'}


def main():
    p = argparse.ArgumentParser(description='View saved point clouds')
    p.add_argument('path', help='.ply/.pcd file or a folder of them')
    p.add_argument('--fps', type=float, default=10.0, help='Playback rate for a folder')
    p.add_argument('--point-size', type=float, default=2.0)
    p.add_argument('--loop', action='store_true', help='Repeat a folder sequence until closed')
    args = p.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        import sys
        raise SystemExit(f'Open3D is not installed for this interpreter ({sys.executable}).\n'
                         f'Install it with:  "{sys.executable}" -m pip install open3d')

    path = Path(args.path)
    if path.is_dir():
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in CLOUD_EXTS)
    elif path.suffix.lower() in CLOUD_EXTS and path.exists():
        files = [path]
    else:
        raise SystemExit(f'No .ply/.pcd found at {path}')
    if not files:
        raise SystemExit(f'No .ply/.pcd files in {path}')

    def load(f):
        cloud = o3d.io.read_point_cloud(str(f))
        if cloud.is_empty():
            raise SystemExit(f'Empty or unreadable point cloud: {f}')
        return cloud

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(f'ZipDepth point cloud - {path.name}', 960, 720)
    vis.get_render_option().point_size = args.point_size
    vis.get_render_option().background_color = np.zeros(3)
    state = {'paused': False, 'quit': False}
    vis.register_key_callback(ord(' '), lambda _: state.update(paused=not state['paused']) or False)
    vis.register_key_callback(ord('Q'), lambda _: state.update(quit=True) or False)
    vis.register_key_callback(256, lambda _: state.update(quit=True) or False)

    cloud = load(files[0])
    vis.add_geometry(cloud)
    print(f'{files[0].name}: {len(cloud.points):,} points'
          + (f'  ({len(files)} frames)' if len(files) > 1 else ''), flush=True)

    def alive():
        running = vis.poll_events()
        vis.update_renderer()
        return running and not state['quit']

    index, period = 0, 1.0 / max(args.fps, 1e-3)
    next_time = time.perf_counter() + period
    while alive():
        if len(files) > 1 and not state['paused'] and time.perf_counter() >= next_time:
            index += 1
            if index >= len(files):
                if not args.loop:
                    state['paused'] = True
                    index = len(files) - 1
                    continue
                index = 0
            frame = load(files[index])
            cloud.points, cloud.colors = frame.points, frame.colors
            vis.update_geometry(cloud)
            next_time = time.perf_counter() + period
        time.sleep(0.005)
    vis.destroy_window()


if __name__ == '__main__':
    main()
