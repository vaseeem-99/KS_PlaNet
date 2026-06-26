"""Create latent-space diagnostics from ``evaluate_pde_rollouts.py`` output."""

from __future__ import absolute_import, division, print_function

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _pca_2d(values):
  centered = values - values.mean(0, keepdims=True)
  _, _, vectors = np.linalg.svd(centered, full_matrices=False)
  return np.dot(centered, vectors[:2].T)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--rollout', required=True)
  parser.add_argument('--output-dir', required=True)
  args = parser.parse_args()

  values = np.load(args.rollout)
  posterior = np.concatenate(
      [values['posterior_mean'], values['posterior_belief']], axis=-1)
  open_loop = np.concatenate(
      [values['open_loop_mean'], values['open_loop_belief']], axis=-1)
  combined = np.concatenate([posterior, open_loop], axis=0)
  projected = _pca_2d(combined)
  post_2d, open_2d = np.split(projected, 2)
  context = int(values['context'])
  os.makedirs(args.output_dir, exist_ok=True)

  fig, axis = plt.subplots(figsize=(7, 6))
  time = np.arange(len(post_2d))
  path = axis.scatter(post_2d[:, 0], post_2d[:, 1], c=time, s=14,
                      cmap='viridis', label='posterior (observed)')
  axis.plot(open_2d[:, 0], open_2d[:, 1], color='#e45756', linewidth=1.0,
            alpha=0.8, label='open-loop')
  axis.scatter(open_2d[context:, 0], open_2d[context:, 1], color='#e45756',
               s=10, alpha=0.65)
  # Matplotlib 3.2 on the TensorFlow container does not accept CSS shorthand.
  axis.axvline(0, color='#dddddd', linewidth=0.5)
  axis.axhline(0, color='#dddddd', linewidth=0.5)
  axis.set_xlabel('PC 1')
  axis.set_ylabel('PC 2')
  axis.set_title('RSSM latent trajectory: posterior vs open-loop')
  axis.legend(loc='best')
  fig.colorbar(path, ax=axis, label='time step')
  fig.tight_layout()
  figure = os.path.join(args.output_dir, 'latent_pca.png')
  fig.savefig(figure, dpi=180)
  plt.close(fig)

  latent_error = np.sqrt(np.mean((posterior - open_loop) ** 2, axis=-1))
  fig, axis = plt.subplots(figsize=(8, 4.5))
  axis.plot(latent_error, color='#e45756', linewidth=1.6)
  axis.axvline(context, color='#333333', linestyle='--', linewidth=1,
               label='open-loop begins')
  axis.set_xlabel('time step')
  axis.set_ylabel('latent RMSE')
  axis.set_title('RSSM posterior/open-loop latent disagreement')
  axis.grid(alpha=0.25)
  axis.legend()
  fig.tight_layout()
  figure = os.path.join(args.output_dir, 'latent_open_loop_error.png')
  fig.savefig(figure, dpi=180)
  plt.close(fig)
  print('Saved latent diagnostics in {}'.format(args.output_dir))


if __name__ == '__main__':
  main()
