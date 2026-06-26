"""Plot image-domain rollout metrics from a real PDE RSSM evaluation archive."""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


TRUTH = '#0072b2'
PREDICTION = '#d55e00'
ACCENT = '#009e73'
GRID = '#cccccc'


def _correlation(prediction, truth):
  values = []
  for pred, real in zip(prediction, truth):
    pred = pred.ravel() - pred.mean()
    real = real.ravel() - real.mean()
    values.append(float(np.dot(pred, real) /
                        (np.linalg.norm(pred) * np.linalg.norm(real) + 1e-12)))
  return np.asarray(values)


def _spectrum(frame):
  field = frame.mean(axis=-1)
  amplitude = np.abs(np.fft.fftshift(np.fft.fft2(field)))
  height, width = amplitude.shape
  y, x = np.indices(amplitude.shape)
  radius = np.sqrt((x - width // 2) ** 2 + (y - height // 2) ** 2).astype(np.int32)
  return np.asarray([
      amplitude[radius == k].mean() for k in range(min(height, width) // 2)])


def _save(figure, path):
  figure.tight_layout()
  figure.savefig(path, dpi=180)
  plt.close(figure)
  print('Saved {}'.format(path))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--rollout', required=True)
  parser.add_argument('--output-dir', required=True)
  parser.add_argument('--dataset', default='ks', choices=['ks', 'kolmogorov'])
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)
  archive = np.load(args.rollout)
  prediction = archive['prediction']
  truth = archive['truth']
  context = int(archive['context'])
  horizon = min(len(prediction), len(truth))
  prediction, truth = prediction[:horizon], truth[:horizon]
  mse = ((prediction - truth) ** 2).reshape(horizon, -1).mean(axis=1)
  corr = _correlation(prediction, truth)
  energy_truth = (truth ** 2).reshape(horizon, -1).mean(axis=1)
  energy_prediction = (prediction ** 2).reshape(horizon, -1).mean(axis=1)

  frames = sorted(set([0, context - 1, context, min(context + 10, horizon - 1), horizon - 1]))
  figure, axes = plt.subplots(2, len(frames), figsize=(3 * len(frames), 5.5))
  for col, step in enumerate(frames):
    axes[0, col].imshow(truth[step])
    axes[1, col].imshow(prediction[step])
    axes[0, col].set_title('t={}'.format(step))
    for axis in (axes[0, col], axes[1, col]):
      axis.set_xticks([])
      axis.set_yticks([])
  axes[0, 0].set_ylabel('truth')
  axes[1, 0].set_ylabel('open-loop prediction')
  _save(figure, os.path.join(
      args.output_dir, 'rollout_strips_{}.png'.format(args.dataset)))

  figure, axis = plt.subplots(figsize=(8, 4.5))
  axis.plot(corr, color=TRUTH, label='Pearson correlation')
  axis.axvline(context, color='#333333', linestyle='--', label='open-loop begins')
  axis.set(xlabel='rollout step', ylabel='correlation', title='Prediction correlation over time')
  axis.grid(alpha=0.35, color=GRID)
  axis.legend()
  _save(figure, os.path.join(args.output_dir, 'correlation_decay.png'))

  figure, axis = plt.subplots(figsize=(8, 4.5))
  axis.semilogy(mse + 1e-12, color=PREDICTION)
  axis.axvline(context, color='#333333', linestyle='--')
  axis.set(xlabel='rollout step', ylabel='MSE (rendered image)', title='Open-loop prediction error')
  axis.grid(alpha=0.35, color=GRID, which='both')
  _save(figure, os.path.join(args.output_dir, 'mse_over_horizon.png'))

  figure, axis = plt.subplots(figsize=(8, 4.5))
  axis.plot(energy_truth, color=TRUTH, label='truth')
  axis.plot(energy_prediction, color=PREDICTION, label='prediction')
  axis.axvline(context, color='#333333', linestyle='--')
  axis.set(xlabel='rollout step', ylabel='mean squared pixel value', title='Rendered-field energy proxy')
  axis.grid(alpha=0.35, color=GRID)
  axis.legend()
  _save(figure, os.path.join(args.output_dir, 'energy_over_horizon.png'))

  middle = min(max(context + 10, 0), horizon - 1)
  truth_spectrum = _spectrum(truth[middle])
  prediction_spectrum = _spectrum(prediction[middle])
  figure, axis = plt.subplots(figsize=(8, 4.5))
  axis.semilogy(truth_spectrum[1:], color=TRUTH, label='truth')
  axis.semilogy(prediction_spectrum[1:], color=PREDICTION, label='prediction')
  axis.set(xlabel='radial image frequency', ylabel='amplitude', title='Rendered-field frequency spectrum')
  axis.grid(alpha=0.35, color=GRID, which='both')
  axis.legend()
  _save(figure, os.path.join(args.output_dir, 'spectral_comparison.png'))

  figure, axis = plt.subplots(figsize=(8, 4.5))
  bins = np.linspace(0, 1, 60)
  axis.hist(truth.ravel(), bins=bins, density=True, alpha=0.55, color=TRUTH, label='truth')
  axis.hist(prediction.ravel(), bins=bins, density=True, alpha=0.55, color=PREDICTION, label='prediction')
  axis.set(xlabel='normalised rendered pixel', ylabel='density', title='Rendered-field value distribution')
  axis.legend()
  _save(figure, os.path.join(args.output_dir, 'value_distribution.png'))

  metrics = {
      'checkpoint_rollout_file': os.path.basename(args.rollout),
      'context_frames': context,
      'horizon_frames': horizon,
      'mse_mean': float(mse.mean()),
      'mse_open_loop_mean': float(mse[context:].mean()),
      'correlation_mean': float(corr.mean()),
      'correlation_open_loop_mean': float(corr[context:].mean()),
      'correlation_final': float(corr[-1]),
  }
  with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as file_:
    json.dump(metrics, file_, indent=2, sort_keys=True)
  print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
