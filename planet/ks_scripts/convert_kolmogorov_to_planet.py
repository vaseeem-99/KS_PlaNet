"""
convert_kolmogorov_to_planet.py
================================
Place at: planet/ks_scripts/convert_kolmogorov_to_planet.py

Converts the Kolmogorov 2D flow dataset into per-episode NPZ files for PlaNet.

Source data (in planet/ks_dataset/):
    X_train.npy  Y_train.npy   -- shape (140000, 1, 128, 128) float32
    X_test.npy   Y_test.npy    -- shape ( 30000, 1, 128, 128) float32
    X_val.npy    Y_val.npy     -- shape ( 30000, 1, 128, 128) float32

The "1" dimension is channels (single-channel vorticity field), NOT time.
Each X[i] / Y[i] is a snapshot pair (u(t), u(t+dt)).

Two scenarios are auto-detected:

  A) SEQUENTIAL: if X[i+1] == Y[i] across consecutive indices, the data is
     one long trajectory split into pairs. Rebuild trajectories of length
     TRAJ_LENGTH (default 50).

  B) INDEPENDENT PAIRS: pairs are 2-frame mini-episodes. PlaNet learns
     one-step dynamics (batch L=2).

Output (in planet/ks_dataset/):
    kolmogorov_train_episodes/
    kolmogorov_test_episodes/
    kolmogorov_val_episodes/

Each NPZ contains:
    image  : (T, 64, 64, 3) uint8     -- vorticity rendered via RdBu_r
    action : (T, 1)         float32   -- zeros
    reward : (T,)           float32   -- zeros

Usage (from planet/):
    python ks_scripts/convert_kolmogorov_to_planet.py

"""

from __future__ import absolute_import, division, print_function

import os
import sys
import time
import traceback
import multiprocessing as mp

import numpy as np

# ── Normalisation (covers ~99.9% of values in [-7, +7]) ──────────────────────
U_MIN = -7.5
U_MAX =  7.5

IMAGE_HEIGHT = 64
IMAGE_WIDTH  = 64


def save_preview_images(episode_dir, figures_dir, prefix, max_frames=8):
    """Save converted image frames from the first episode as PNG outputs."""
    files = sorted(f for f in os.listdir(episode_dir) if f.endswith('.npz'))
    if not files:
        print(f'No preview written for {prefix}; no episodes found.')
        return []

    os.makedirs(figures_dir, exist_ok=True)
    frame_dir = os.path.join(figures_dir, prefix + '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    path = os.path.join(episode_dir, files[0])
    with np.load(path) as episode:
        all_frames = np.asarray(episode['image'])
    frame_indices = np.linspace(
        0, len(all_frames) - 1, min(max_frames, len(all_frames))).astype(int)
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

# ── Conversion caps -- override at top of script ─────────────────────────────
# Set to None to convert every pair.
# Kolmogorov has 140k train + 30k val + 30k test pairs. 20k train pairs
# is more than enough for a 128->64 rendered world model.
MAX_TRAIN_PAIRS = 20000
MAX_TEST_PAIRS  =  5000
MAX_VAL_PAIRS   =  5000

# If detected as sequential, group this many consecutive snapshots per trajectory.
TRAJ_LENGTH = 50


# ─────────────────────────────────────────────────────────────────────────────
# Sequential vs independent detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_sequential(X, Y, n_samples=20, atol=1e-3):
    """Check whether Y[i] == X[i+1] across a sample of indices."""
    n = len(X)
    if n < 2:
        return False, np.inf
    idxs = np.linspace(0, n - 2, min(n_samples, n - 1)).astype(int)
    diffs = []
    for i in idxs:
        y_i      = np.asarray(Y[i]).ravel()
        x_next   = np.asarray(X[i + 1]).ravel()
        denom    = np.abs(y_i).mean() + 1e-6
        rel_diff = np.abs(y_i - x_next).mean() / denom
        diffs.append(rel_diff)
    mean_diff = float(np.mean(diffs))
    return mean_diff < atol, mean_diff


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _downsample(field_128, target_h=IMAGE_HEIGHT, target_w=IMAGE_WIDTH):
    """Downsample 128x128 -> 64x64 by 2x2 mean pooling."""
    h, w = field_128.shape
    if h == target_h and w == target_w:
        return field_128
    fh, fw = h // target_h, w // target_w
    return field_128[:fh * target_h, :fw * target_w] \
        .reshape(target_h, fh, target_w, fw).mean(axis=(1, 3))


def _render_frame(field_2d, cmap):
    """A single 128x128 field -> (64, 64, 3) uint8 via colormap."""
    field_64 = _downsample(field_2d)
    norm     = np.clip((field_64 - U_MIN) / (U_MAX - U_MIN), 0.0, 1.0)
    rgba     = cmap(norm)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _frames_to_episode(frame_list, cmap_name='RdBu_r'):
    """List of 2D fields -> PlaNet episode dict."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm
    cmap   = cm.get_cmap(cmap_name)
    frames = np.stack([_render_frame(f, cmap) for f in frame_list])
    T      = frames.shape[0]
    return {
        'image':  frames,
        'action': np.zeros((T, 1), dtype=np.float32),
        'reward': np.zeros(T,      dtype=np.float32),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args):
    idx, frame_list, out_path, cmap_name = args
    try:
        episode = _frames_to_episode(frame_list, cmap_name)
        np.savez_compressed(out_path, **episode)
        return idx, True, None
    except Exception:
        return idx, False, traceback.format_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Episode builders
# ─────────────────────────────────────────────────────────────────────────────

def build_pair_episodes(X, Y, max_pairs):
    """Each (X[i], Y[i]) becomes a 2-frame episode."""
    n = min(len(X), max_pairs) if max_pairs else len(X)
    for i in range(n):
        x_field = np.asarray(X[i])[0]   # strip channel axis (1, 128, 128) -> (128, 128)
        y_field = np.asarray(Y[i])[0]
        yield i, [x_field, y_field]


def build_trajectory_episodes(X, Y, max_pairs, traj_length):
    """Group consecutive sequential snapshots into trajectories.

    For tiny dry-runs where n_pairs < traj_length, emit one short trajectory
    instead of silently producing zero episodes.
    """
    n_pairs = min(len(X), max_pairs) if max_pairs else len(X)
    for i, start in enumerate(range(0, n_pairs, traj_length)):
        end    = min(start + traj_length, n_pairs)
        if end <= start:
            continue
        frames = [np.asarray(X[j])[0] for j in range(start, end)]
        frames.append(np.asarray(Y[end - 1])[0])
        yield i, frames


# ─────────────────────────────────────────────────────────────────────────────
# Split conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert_split(X, Y, out_dir, split_name, max_pairs, is_sequential,
                  traj_length, workers=4):
    os.makedirs(out_dir, exist_ok=True)

    if is_sequential:
        ep_builder = build_trajectory_episodes(X, Y, max_pairs, traj_length)
        n_pairs = min(len(X), max_pairs) if max_pairs else len(X)
        total_eps = int(np.ceil(float(n_pairs) / traj_length))
        ep_kind = f'trajectories (up to T={traj_length + 1})'
    else:
        ep_builder = build_pair_episodes(X, Y, max_pairs)
        total_eps  = min(len(X), max_pairs) if max_pairs else len(X)
        ep_kind    = '2-frame pairs'

    print(f'\n[{split_name}] {total_eps} episodes as {ep_kind} -> {out_dir}')

    tasks   = []
    skipped = 0
    for idx, frames in ep_builder:
        out_path = os.path.join(out_dir, f'episode_{idx:06d}.npz')
        if os.path.exists(out_path):
            skipped += 1
            continue
        tasks.append((idx, frames, out_path, 'RdBu_r'))
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
            if done % 500 == 0 or done == total_eps:
                elapsed = time.time() - t_start
                rate    = (ok_count + fail_count) / max(elapsed, 1)
                pct     = 100 * done / max(total_eps, 1)
                print(f'  [{split_name}] {done}/{total_eps} ({pct:.0f}%)  '
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
                if done % 500 == 0 or done == total_eps:
                    elapsed = time.time() - t_start
                    rate    = (ok_count + fail_count) / max(elapsed, 1)
                    pct     = 100 * done / max(total_eps, 1)
                    print(f'  [{split_name}] {done}/{total_eps} ({pct:.0f}%)  '
                          f'{ok_count} ok  {fail_count} failed  {rate:.1f} ep/s')

    elapsed = time.time() - t_start

    print(f'[{split_name}] Done: {ok_count + skipped}/{total_eps} OK  '
          f'{fail_count} failed  ({elapsed:.1f}s)\n')
    return {'total': total_eps, 'converted': ok_count, 'skipped': skipped,
            'failed': fail_count, 'failures': failures, 'elapsed_s': elapsed}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(script_dir, '..', 'ks_dataset')
    figures_dir = os.path.join(data_dir, 'figures')
    workers    = int(os.environ.get('CONVERT_WORKERS', min(mp.cpu_count(), 8)))

    splits = {
        'train': ('X_train.npy', 'Y_train.npy', 'kolmogorov_train_episodes', MAX_TRAIN_PAIRS),
        'val':   ('X_val.npy',   'Y_val.npy',   'kolmogorov_val_episodes',   MAX_VAL_PAIRS),
        'test':  ('X_test.npy',  'Y_test.npy',  'kolmogorov_test_episodes',  MAX_TEST_PAIRS),
    }

    for split_name, (xf, yf, _, _) in splits.items():
        for fname in (xf, yf):
            path = os.path.join(data_dir, fname)
            if not os.path.exists(path):
                print(f'ERROR: source file not found: {path}')
                sys.exit(1)

    loaded = {}
    for split_name, (xf, yf, out_subdir, max_pairs) in splits.items():
        print(f'Loading {xf} (mmap) ...')
        X = np.load(os.path.join(data_dir, xf), mmap_mode='r')
        print(f'Loading {yf} (mmap) ...')
        Y = np.load(os.path.join(data_dir, yf), mmap_mode='r')
        print(f'  X: {X.shape}  Y: {Y.shape}')
        loaded[split_name] = (X, Y, out_subdir, max_pairs)

    X_train, Y_train = loaded['train'][0], loaded['train'][1]
    print('\nChecking whether data is sequential (Y[i] == X[i+1])...')
    is_seq, mean_diff = detect_sequential(X_train, Y_train, n_samples=20)
    print(f'  Mean relative diff between Y[i] and X[i+1]: {mean_diff:.4f}')
    print(f'  Layout: {"SEQUENTIAL (rebuilding trajectories)" if is_seq else "INDEPENDENT PAIRS"}')

    print('Conversion config:')
    print('  dataset             : kolmogorov')
    print(f'  train samples       : {int(X_train.shape[0])}')
    print(f'  test samples        : {int(loaded["test"][0].shape[0])}')
    print(f'  val samples         : {int(loaded["val"][0].shape[0])}')
    print(f'  native resolution   : {int(X_train.shape[-1])}')
    print(f'  rendered resolution : {IMAGE_HEIGHT}')
    print(f'  sequential          : {bool(is_seq)}')
    print(f'  sequential diff     : {float(mean_diff):.6f}')
    print(f'  trajectory length   : {TRAJ_LENGTH if is_seq else 2}')
    print(f'  workers             : {workers}')

    all_stats = {}
    for split_name, (X, Y, out_subdir, max_pairs) in loaded.items():
        out_dir = os.path.join(data_dir, out_subdir)
        stats = convert_split(X, Y, out_dir, split_name,
                              max_pairs=max_pairs,
                              is_sequential=is_seq,
                              traj_length=TRAJ_LENGTH,
                              workers=workers)
        all_stats[split_name] = stats

    summary = {}
    total_failures = 0
    total_bytes    = 0
    for split_name, (_, _, out_subdir, _) in loaded.items():
        out_dir = os.path.join(data_dir, out_subdir)
        n_files = len([f for f in os.listdir(out_dir) if f.endswith('.npz')])
        nbytes  = sum(os.path.getsize(os.path.join(out_dir, f))
                      for f in os.listdir(out_dir) if f.endswith('.npz'))
        s = all_stats[split_name]
        summary[f'kolmogorov_{split_name}/files_on_disk'] = n_files
        summary[f'kolmogorov_{split_name}/disk_gb']       = nbytes / 1e9
        summary[f'kolmogorov_{split_name}/failed']        = s['failed']
        summary[f'kolmogorov_{split_name}/converted']     = s['converted']
        summary[f'kolmogorov_{split_name}/elapsed_s']     = s['elapsed_s']
        total_failures += s['failed']
        total_bytes    += nbytes

    summary['kolmogorov/total_failures']    = total_failures
    summary['kolmogorov/total_disk_gb']     = total_bytes / 1e9
    summary['kolmogorov/conversion_status'] = 'PASSED' if total_failures == 0 else 'FAILED'
    if total_failures > 0:
        print(f'\nKolmogorov conversion complete -- {total_failures} failures.')
    else:
        print('\nKolmogorov conversion complete -- 0 failures.')

    print('=' * 60)
    print(f'kolmogorov_train_episodes/  {summary.get("kolmogorov_train/files_on_disk", 0)} files')
    print(f'kolmogorov_val_episodes/    {summary.get("kolmogorov_val/files_on_disk", 0)} files')
    print(f'kolmogorov_test_episodes/   {summary.get("kolmogorov_test/files_on_disk", 0)} files')
    print(f'Total disk: {total_bytes/1e9:.2f} GB')
    for split_name, (_, _, out_subdir, _) in loaded.items():
        save_preview_images(
            os.path.join(data_dir, out_subdir),
            figures_dir,
            f'kolmogorov_{split_name}')
    print('Summary:')
    for key in sorted(summary):
        print(f'  {key}: {summary[key]}')
    print()
    print('Next: train PlaNet on Kolmogorov with:')
    print('  python -m planet.scripts.train \\')
    print('    --logdir ks_dataset/kolmogorov_run \\')
    print('    --config kolmogorov_config \\')
    print("    --params '{tasks: [kolmogorov_flow]}'")


if __name__ == '__main__':
    mp.freeze_support()
    main()
