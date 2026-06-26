"""
convert_ks_to_planet.py
=======================
Place at: planet/ks_scripts/convert_ks_to_planet.py

Converts the two bulk KS NPZ files into per-episode NPZ files that
PlaNet's data loader expects. Each observation image is a rolling
space-time window with time on rows and space on columns.

Usage (run from planet/ directory):
    python ks_scripts/convert_ks_to_planet.py

Reads from:  ks_dataset/ks_train.npz  ks_dataset/ks_test.npz
Writes to:   ks_dataset/train_episodes/  ks_dataset/test_episodes/

"""

from __future__ import absolute_import, division, print_function

import os
import sys
import time
import traceback
import multiprocessing as mp

import numpy as np

U_MIN = -3.7
U_MAX =  3.7
IMAGE_HEIGHT = 64
IMAGE_WIDTH  = 64


def save_preview_images(episode_dir, figures_dir, prefix, max_frames=8):
    """Save PlaNet input frames from the first episode as PNG outputs."""
    files = sorted(
        f for f in os.listdir(episode_dir) if f.endswith('.npz'))
    if not files:
        print(f'No preview written for {prefix}; no episodes found.')
        return []

    os.makedirs(figures_dir, exist_ok=True)
    frame_dir = os.path.join(figures_dir, prefix + '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    path = os.path.join(episode_dir, files[0])
    with np.load(path) as episode:
        all_frames = np.asarray(episode['image'])
    start_index = min(IMAGE_HEIGHT - 1, len(all_frames) - 1)
    frame_indices = np.linspace(
        start_index, len(all_frames) - 1,
        min(max_frames, len(all_frames) - start_index)).astype(int)
    frames = all_frames[frame_indices]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    outputs = []
    for index, frame in zip(frame_indices, frames):
        out_path = os.path.join(frame_dir, f'frame_{index:03d}.png')
        plt.imsave(out_path, frame)
        outputs.append(out_path)

    strip = np.concatenate(frames, axis=1)
    strip_path = os.path.join(figures_dir, prefix + '_preview_strip.png')
    plt.imsave(strip_path, strip)
    outputs.append(strip_path)
    print(f'Preview images written for {prefix}: {strip_path}')
    return outputs


def save_space_time_previews(
        u_array, figures_dir, prefix, L_array=None, dt_array=None,
        max_trajectories=8):
    """Save full KS space-time trajectory plots for visual inspection."""
    os.makedirs(figures_dir, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    count = min(max_trajectories, len(u_array))
    outputs = []
    ncols = min(4, count)
    nrows = int(np.ceil(float(count) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False, figsize=(4 * ncols, 3 * nrows))

    for plot_index in range(count):
        row, col = divmod(plot_index, ncols)
        ax = axes[row][col]
        traj = np.asarray(u_array[plot_index])
        L = float(L_array[plot_index]) if L_array is not None else traj.shape[1]
        dt = float(dt_array[plot_index]) if dt_array is not None else 1.0
        extent = [0, L, 0, traj.shape[0] * dt]
        ax.imshow(
            traj, aspect='auto', origin='lower', cmap='RdBu_r',
            vmin=U_MIN, vmax=U_MAX, extent=extent)
        ax.set_title(f'Traj {plot_index}  L={L:.1f}', fontsize=8)
        ax.set_xlabel('x')
        ax.set_ylabel('t')

        out_path = os.path.join(
            figures_dir, f'{prefix}_trajectory_{plot_index:03d}.png')
        plt.imsave(
            out_path, traj, cmap='RdBu_r', vmin=U_MIN, vmax=U_MAX,
            origin='lower')
        outputs.append(out_path)

    for empty_index in range(count, nrows * ncols):
        row, col = divmod(empty_index, ncols)
        axes[row][col].axis('off')

    fig.suptitle(f'{prefix}: full KS space-time trajectories')
    fig.tight_layout()
    grid_path = os.path.join(figures_dir, f'{prefix}_trajectory_grid.png')
    fig.savefig(grid_path, dpi=150)
    plt.close(fig)
    outputs.append(grid_path)
    print(f'Full space-time trajectory preview written: {grid_path}')
    return outputs


def _render_space_time_frame(u_traj, t, cmap):
    """Render a rolling KS space-time window ending at time index t.

    Rows are time, columns are space. For early time steps, the missing
    history is padded with the first available row.
    """
    spatial_idx = np.linspace(0, u_traj.shape[1] - 1, IMAGE_WIDTH).astype(int)
    start = max(0, t - IMAGE_HEIGHT + 1)
    window = u_traj[start:t + 1, :][:, spatial_idx]
    if len(window) < IMAGE_HEIGHT:
        pad = np.repeat(window[:1], IMAGE_HEIGHT - len(window), axis=0)
        window = np.concatenate([pad, window], axis=0)
    norm = np.clip((window - U_MIN) / (U_MAX - U_MIN), 0.0, 1.0)
    rgba = cmap(norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def _trajectory_to_episode(u_traj, cmap_name='RdBu_r'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    cmap   = cm.get_cmap(cmap_name)
    T      = u_traj.shape[0]
    frames = np.stack([
        _render_space_time_frame(u_traj, t, cmap) for t in range(T)])
    return {
        'image':  frames,
        'action': np.zeros((T, 1), dtype=np.float32),
        'reward': np.zeros(T,      dtype=np.float32),
    }


def _worker(args):
    idx, u_traj, out_path, cmap_name = args
    try:
        episode = _trajectory_to_episode(u_traj, cmap_name)
        np.savez_compressed(out_path, **episode)
        return idx, True, None
    except Exception:
        return idx, False, traceback.format_exc()


def convert_dataset(u_array, out_dir, split_name, cmap_name='RdBu_r',
                    workers=4):
    os.makedirs(out_dir, exist_ok=True)
    N = u_array.shape[0]
    print(f'\n[{split_name}] Converting {N} trajectories '
          f'shape={u_array.shape} -> {out_dir}')

    tasks   = []
    skipped = 0
    for i in range(N):
        out_path = os.path.join(out_dir, f'episode_{i:05d}.npz')
        if os.path.exists(out_path):
            skipped += 1
            continue
        tasks.append((i, u_array[i], out_path, cmap_name))
    if skipped:
        print(f'  Skipping {skipped} already-converted episodes.')

    ok_count   = 0
    fail_count = 0
    failures   = []
    t_start    = time.time()

    if tasks and workers <= 1:
        results = map(_worker, tasks)
        for idx, success, err in results:
            if success:
                ok_count += 1
            else:
                fail_count += 1
                err_type = 'UnknownError'
                if err:
                    last_line = [l for l in err.strip().splitlines() if l][-1]
                    err_type  = last_line.split(':')[0].strip()
                failures.append({
                    'split': split_name, 'episode_idx': idx,
                    'error_type': err_type, 'error_message': err or 'unknown',
                })
                print(f'  [FAIL] {split_name} episode {idx}: {err_type}')
                if err:
                    print('  ' + err.replace('\n', '\n  '))

            done = skipped + ok_count + fail_count
            if done % 200 == 0 or done == N:
                elapsed = time.time() - t_start
                rate    = (ok_count + fail_count) / max(elapsed, 1)
                pct     = 100 * done / N
                print(f'  [{split_name}] {done}/{N} ({pct:.0f}%)  '
                      f'{ok_count} ok  {fail_count} failed  {rate:.1f} ep/s')

    elif tasks:
        with mp.Pool(workers) as pool:
            for idx, success, err in pool.imap_unordered(_worker, tasks):
                if success:
                    ok_count += 1
                else:
                    fail_count += 1
                    err_type = 'UnknownError'
                    if err:
                        last_line = [l for l in err.strip().splitlines() if l][-1]
                        err_type  = last_line.split(':')[0].strip()
                    failures.append({
                        'split': split_name, 'episode_idx': idx,
                        'error_type': err_type, 'error_message': err or 'unknown',
                    })
                    print(f'  [FAIL] {split_name} episode {idx}: {err_type}')
                    if err:
                        print('  ' + err.replace('\n', '\n  '))

                done = skipped + ok_count + fail_count
                if done % 200 == 0 or done == N:
                    elapsed = time.time() - t_start
                    rate    = (ok_count + fail_count) / max(elapsed, 1)
                    pct     = 100 * done / N
                    print(f'  [{split_name}] {done}/{N} ({pct:.0f}%)  '
                          f'{ok_count} ok  {fail_count} failed  {rate:.1f} ep/s')

    elapsed = time.time() - t_start
    print(f'[{split_name}] Done: {ok_count + skipped}/{N} OK  '
          f'{fail_count} failed  ({elapsed:.1f}s)\n')
    return {'total': N, 'converted': ok_count, 'skipped': skipped,
            'failed': fail_count, 'failures': failures, 'elapsed_s': elapsed}


def main():
    # Script is in ks_scripts/, data is in ks_dataset/ -- go up one level.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(script_dir, '..', 'ks_dataset')
    train_npz  = os.path.join(data_dir, 'ks_train.npz')
    test_npz   = os.path.join(data_dir, 'ks_test.npz')
    train_out  = os.path.join(data_dir, 'train_episodes')
    test_out   = os.path.join(data_dir, 'test_episodes')
    figures_dir = os.path.join(data_dir, 'figures')
    workers    = int(os.environ.get('CONVERT_WORKERS', min(mp.cpu_count(), 8)))

    for path in (train_npz, test_npz):
        if not os.path.exists(path):
            print(f'ERROR: source file not found: {path}')
            sys.exit(1)

    print(f'Loading {train_npz} (memory-mapped) ...')
    train_data = np.load(train_npz, mmap_mode='r')
    u_train    = train_data['u']
    L_train    = train_data['L'] if 'L' in train_data.files else None
    dt_train   = train_data['dt'] if 'dt' in train_data.files else None
    print(f'  train u: {u_train.shape}  dtype={u_train.dtype}')

    print(f'Loading {test_npz} (memory-mapped) ...')
    test_data  = np.load(test_npz, mmap_mode='r')
    u_test     = test_data['u']
    L_test     = test_data['L'] if 'L' in test_data.files else None
    dt_test    = test_data['dt'] if 'dt' in test_data.files else None
    print(f'  test  u: {u_test.shape}  dtype={u_test.dtype}')

    print('Conversion config:')
    print(f'  dataset            : ks')
    print(f'  train trajectories : {int(u_train.shape[0])}')
    print(f'  train timesteps    : {int(u_train.shape[1])}')
    print(f'  test trajectories  : {int(u_test.shape[0])}')
    print(f'  test timesteps     : {int(u_test.shape[1])}')
    print(f'  spatial points     : {int(u_train.shape[2])}')
    print(f'  image size         : {IMAGE_HEIGHT}x{IMAGE_WIDTH}')
    print(f'  rendering          : rolling space-time windows')
    print(f'  workers            : {workers}')

    train_stats = convert_dataset(u_train, train_out, 'train', workers=workers)
    test_stats  = convert_dataset(u_test,  test_out,  'test',  workers=workers)

    n_train_files = len([f for f in os.listdir(train_out) if f.endswith('.npz')])
    n_test_files  = len([f for f in os.listdir(test_out)  if f.endswith('.npz')])
    train_bytes = sum(os.path.getsize(os.path.join(train_out, f))
                      for f in os.listdir(train_out) if f.endswith('.npz'))
    test_bytes  = sum(os.path.getsize(os.path.join(test_out, f))
                      for f in os.listdir(test_out)  if f.endswith('.npz'))
    total_failures = train_stats['failed'] + test_stats['failed']

    summary = {
        'ks_train/files_on_disk': n_train_files,
        'ks_train/disk_gb':       train_bytes / 1e9,
        'ks_train/failed':        train_stats['failed'],
        'ks_test/files_on_disk':  n_test_files,
        'ks_test/disk_gb':        test_bytes / 1e9,
        'ks_test/failed':         test_stats['failed'],
        'ks/total_failures':      total_failures,
        'ks/total_disk_gb':       (train_bytes + test_bytes) / 1e9,
        'ks/conversion_status':   'PASSED' if total_failures == 0 else 'FAILED',
    }

    if total_failures > 0:
        print(f'\nKS conversion complete -- {total_failures} failures.')
    else:
        print('\nKS conversion complete -- 0 failures.')

    print('=' * 60)
    print(f'train_episodes/  {n_train_files} files  ({train_bytes/1e9:.2f} GB)')
    print(f'test_episodes/   {n_test_files} files  ({test_bytes/1e9:.2f} GB)')
    save_space_time_previews(
        u_train, figures_dir, 'ks_train', L_train, dt_train)
    save_space_time_previews(
        u_test, figures_dir, 'ks_test', L_test, dt_test)
    save_preview_images(train_out, figures_dir, 'ks_train')
    save_preview_images(test_out, figures_dir, 'ks_test')
    print('Summary:')
    for key in sorted(summary):
        print(f'  {key}: {summary[key]}')


if __name__ == '__main__':
    mp.freeze_support()
    main()
