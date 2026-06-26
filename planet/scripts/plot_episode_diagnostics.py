"""Create PDE/input diagnostics from converted PlaNet episode NPZ files.

These plots are for checking the data that will be fed to the convolutional
encoder. They do not require a trained checkpoint.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CMAP = 'RdBu_r'


def load_images(episode_dir, max_episodes):
  files = sorted(glob.glob(os.path.join(episode_dir, '*.npz')))[:max_episodes]
  episodes = []
  for filename in files:
    with np.load(filename) as data:
      episodes.append(data['image'].astype(np.float32) / 255.0)
  return files, episodes


def scalar_field(image):
  if image.ndim == 3:
    return image.mean(-1)
  return image


def save_frame_grid(episodes, out_path, title, max_frames=8):
  if not episodes:
    return
  count = len(episodes)
  fig, axes = plt.subplots(count, max_frames, figsize=(2 * max_frames, 2 * count))
  if count == 1:
    axes = axes[None]
  for row, ep in enumerate(episodes):
    idxs = np.linspace(0, len(ep) - 1, min(max_frames, len(ep))).astype(int)
    for col in range(max_frames):
      ax = axes[row][col]
      ax.axis('off')
      if col >= len(idxs):
        continue
      idx = idxs[col]
      ax.imshow(ep[idx], origin='lower')
      ax.set_title('t={}'.format(idx), fontsize=8)
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)


def save_episode_strip(episodes, out_path, title, max_frames=12):
  if not episodes:
    return
  ep = episodes[0]
  idxs = np.linspace(0, len(ep) - 1, min(max_frames, len(ep))).astype(int)
  strip = np.concatenate(ep[idxs], axis=1)
  plt.imsave(out_path, strip)

  fig, ax = plt.subplots(figsize=(max(8, len(idxs) * 1.2), 2.5))
  ax.imshow(strip, origin='lower')
  ax.set_title(title)
  ax.set_xticks([])
  ax.set_yticks([])
  fig.tight_layout()
  fig.savefig(out_path.replace('.png', '_labeled.png'), dpi=150)
  plt.close(fig)


def save_temporal_stats(episodes, out_path, title):
  if not episodes:
    return
  ep = episodes[0]
  frames = np.asarray([scalar_field(frame) for frame in ep])
  mean_value = frames.reshape(len(frames), -1).mean(axis=1)
  std_value = frames.reshape(len(frames), -1).std(axis=1)
  delta = np.abs(frames[1:] - frames[:-1]).reshape(len(frames) - 1, -1).mean(axis=1)

  fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
  axes[0].plot(mean_value)
  axes[0].set_ylabel('mean')
  axes[1].plot(std_value)
  axes[1].set_ylabel('std')
  axes[2].plot(delta)
  axes[2].set_ylabel('|delta|')
  axes[2].set_xlabel('time index')
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)


def save_spectrum(episodes, out_path, title):
  if not episodes:
    return
  ep = episodes[0]
  frames = np.asarray([scalar_field(frame) for frame in ep])
  spectra = np.abs(np.fft.rfft(frames.reshape(len(frames), -1), axis=1))
  spectrum = spectra.mean(axis=0)
  k = np.arange(len(spectrum))

  fig, ax = plt.subplots(figsize=(8, 4))
  ax.semilogy(k[1:], spectrum[1:] + 1e-12)
  ax.set_xlabel('flattened spatial frequency index')
  ax.set_ylabel('mean amplitude')
  ax.set_title(title)
  ax.grid(True, alpha=0.25)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)


def save_dataset_mosaic(episodes, out_path, title):
  if not episodes:
    return
  count = len(episodes)
  ncols = min(4, count)
  nrows = int(np.ceil(float(count) / ncols))
  fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
  axes = np.asarray(axes).reshape(nrows, ncols)
  for i, ep in enumerate(episodes):
    row, col = divmod(i, ncols)
    ax = axes[row][col]
    mid = len(ep) // 2
    ax.imshow(ep[mid], origin='lower')
    ax.set_title('episode {} t={}'.format(i, mid), fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
  for i in range(count, nrows * ncols):
    row, col = divmod(i, ncols)
    axes[row][col].axis('off')
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(out_path, dpi=150)
  plt.close(fig)


def plot_split(dataset, split, episode_dir, outdir, max_episodes):
  files, episodes = load_images(episode_dir, max_episodes)
  if not episodes:
    print('No episodes found:', episode_dir)
    return
  prefix = '{}_{}'.format(dataset, split)
  print('Plotting {} episodes from {}'.format(len(episodes), episode_dir))
  save_frame_grid(
      episodes,
      os.path.join(outdir, prefix + '_frame_grid.png'),
      '{} {} PlaNet input frames'.format(dataset.upper(), split))
  save_episode_strip(
      episodes,
      os.path.join(outdir, prefix + '_episode_strip.png'),
      '{} {} first episode strip'.format(dataset.upper(), split))
  save_temporal_stats(
      episodes,
      os.path.join(outdir, prefix + '_temporal_stats.png'),
      '{} {} first episode temporal stats'.format(dataset.upper(), split))
  save_spectrum(
      episodes,
      os.path.join(outdir, prefix + '_spectrum.png'),
      '{} {} first episode spectrum'.format(dataset.upper(), split))
  save_dataset_mosaic(
      episodes,
      os.path.join(outdir, prefix + '_episode_mosaic.png'),
      '{} {} episode mosaic'.format(dataset.upper(), split))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', required=True, choices=['ks', 'kolmogorov'])
  parser.add_argument('--data-dir', default='ks_dataset')
  parser.add_argument('--outdir', default=None)
  parser.add_argument('--max-episodes', type=int, default=8)
  args = parser.parse_args()

  if args.outdir is None:
    args.outdir = os.path.join(args.data_dir, 'figures', args.dataset + '_diagnostics')
  os.makedirs(args.outdir, exist_ok=True)

  if args.dataset == 'ks':
    splits = {
        'train': os.path.join(args.data_dir, 'train_episodes'),
        'test': os.path.join(args.data_dir, 'test_episodes'),
    }
  else:
    splits = {
        'train': os.path.join(args.data_dir, 'kolmogorov_train_episodes'),
        'val': os.path.join(args.data_dir, 'kolmogorov_val_episodes'),
        'test': os.path.join(args.data_dir, 'kolmogorov_test_episodes'),
    }

  for split, episode_dir in splits.items():
    plot_split(args.dataset, split, episode_dir, args.outdir, args.max_episodes)
  print('Done. Plots:', args.outdir)


if __name__ == '__main__':
  main()
