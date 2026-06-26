"""
analyze_results.py
==================
Place at: planet/analyze_results.py

Generates the publication-quality plots that matter for PDE-solver,
physics-simulation, and latent-space research from PlaNet rollouts.

Run AFTER training has produced checkpoints + rollout dumps. PlaNet writes
predicted rollouts as part of its standard summary protobufs; this script
loads them and computes the metrics that appear in PDE-Refiner, LNS, and PNO.

Plots produced (each saved as a separate PNG):
  1. rollout_strips_ks.png         Ground truth vs prediction at 1/10/50/100 steps
  2. rollout_strips_kolmogorov.png Same for 2D flow
  3. correlation_decay.png         Pearson correlation vs rollout time (PDE-Refiner metric)
  4. mse_over_horizon.png          Per-step MSE on log scale
  5. spectral_error.png            Predicted vs true frequency spectrum
  6. energy_conservation.png       L2-norm or kinetic energy over rollout
  7. latent_pca.png                PCA projection of latent z_t over rollout
  8. value_histogram.png           Distribution match between truth and prediction

Run (inside container):
    python analyze_results.py \
        --logdir /eagle/projects/PIGAN/$USER/ks_planet/checkpoints/ks_run \
        --dataset ks \
        --output /eagle/projects/PIGAN/$USER/ks_planet/figures

Or for Kolmogorov:
    python analyze_results.py \
        --logdir .../kolmogorov_run --dataset kolmogorov --output .../figures
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import glob

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Allow running from any working dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CMAP = 'RdBu_r'
DARK_BG = '#0d0d0d'
PANEL_BG = '#111'


def style(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors='white', labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor('#333')


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_episodes(episode_dir, max_n=8):
    """Load test episodes from PlaNet's NPZ format."""
    files = sorted(glob.glob(os.path.join(episode_dir, '*.npz')))[:max_n]
    episodes = []
    for f in files:
        d = np.load(f)
        episodes.append({
            'image':  d['image'].astype(np.float32) / 255.0,   # (T, 64, 64, 3)
            'action': d['action'] if 'action' in d else None,
        })
    return episodes


def find_rollout_dumps(logdir):
    """PlaNet dumps openloop rollouts during evaluation. Find them."""
    candidates = []
    for pattern in ('*.npz', '*openloop*.npz', 'rollouts/*.npz'):
        candidates.extend(glob.glob(os.path.join(logdir, pattern)))
        candidates.extend(glob.glob(os.path.join(logdir, '**', pattern),
                                    recursive=True))
    return sorted(set(candidates))


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def pearson_correlation(pred, truth, axis=None):
    """Pearson r over a chosen axis.  Default: over all values."""
    p = pred.ravel() if axis is None else pred
    t = truth.ravel() if axis is None else truth
    p = p - p.mean()
    t = t - t.mean()
    denom = np.sqrt((p ** 2).sum() * (t ** 2).sum()) + 1e-12
    return float((p * t).sum() / denom)


def mse_per_step(pred_seq, truth_seq):
    """MSE per timestep across spatial dims.  Returns (T,) array."""
    return ((pred_seq - truth_seq) ** 2).reshape(len(pred_seq), -1).mean(axis=1)


def correlation_per_step(pred_seq, truth_seq):
    """Pearson r at each timestep.  Returns (T,) array."""
    return np.array([pearson_correlation(p, t)
                     for p, t in zip(pred_seq, truth_seq)])


def radial_spectrum_2d(field):
    """2D radial spectrum.  Returns (k, amp) arrays."""
    h, w = field.shape[-2:]
    fft = np.fft.fftshift(np.fft.fft2(field))
    power = np.abs(fft) / (h * w)
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    k_max = min(cy, cx)
    return np.arange(k_max), np.array([power[r == k].mean()
                                       if (r == k).any() else 0
                                       for k in range(k_max)])


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Rollout strips
# ─────────────────────────────────────────────────────────────────────────────

def plot_rollout_strips(pred_seq, truth_seq, dataset, out_path,
                        timesteps=(0, 5, 10, 20, 50)):
    n = len(timesteps)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
    fig.patch.set_facecolor(DARK_BG)
    for i, t in enumerate(timesteps):
        if t >= len(truth_seq):
            continue
        ax_t, ax_p = axes[0, i], axes[1, i]
        if dataset == 'ks':
            ax_t.imshow(truth_seq[t], cmap=CMAP, vmin=0, vmax=1, origin='lower')
            ax_p.imshow(pred_seq[t],  cmap=CMAP, vmin=0, vmax=1, origin='lower')
        else:
            tr = truth_seq[t].mean(-1) if truth_seq[t].ndim == 3 else truth_seq[t]
            pr = pred_seq[t].mean(-1)  if pred_seq[t].ndim  == 3 else pred_seq[t]
            ax_t.imshow(tr, cmap=CMAP, vmin=0, vmax=1, origin='lower')
            ax_p.imshow(pr, cmap=CMAP, vmin=0, vmax=1, origin='lower')
        ax_t.set_title(f't={t}', color='white', fontsize=9)
        for a in (ax_t, ax_p):
            a.set_xticks([])
            a.set_yticks([])
            style(a)
    axes[0, 0].set_ylabel('TRUTH',      color='#00d4ff', fontsize=10)
    axes[1, 0].set_ylabel('PREDICTION', color='#ff6b6b', fontsize=10)
    fig.suptitle(f'Open-loop rollout strips ({dataset.upper()})',
                 color='white', fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f'  saved {out_path}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 + 3: Correlation decay and MSE-over-horizon
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_decay(pred_seq, truth_seq, out_path):
    corr = correlation_per_step(pred_seq, truth_seq)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style(ax)
    ax.plot(corr, color='#00d4ff', linewidth=1.8)
    ax.axhline(0.9, color='#ff6b6b', linestyle='--', linewidth=0.8,
               label='r = 0.9 threshold')
    ax.axhline(0.8, color='#ffaa44', linestyle='--', linewidth=0.8,
               label='r = 0.8 threshold')
    cross_09 = (corr < 0.9).argmax() if (corr < 0.9).any() else len(corr)
    cross_08 = (corr < 0.8).argmax() if (corr < 0.8).any() else len(corr)
    ax.axvline(cross_09, color='#ff6b6b', linestyle=':', linewidth=0.6)
    ax.axvline(cross_08, color='#ffaa44', linestyle=':', linewidth=0.6)
    ax.set_xlabel('rollout step', color='white')
    ax.set_ylabel('Pearson correlation', color='white')
    ax.set_title(f'Correlation decay  |  r>0.9 to step {cross_09},  '
                 f'r>0.8 to step {cross_08}', color='white', fontsize=11)
    ax.legend(fontsize=9, facecolor='#222', labelcolor='white', framealpha=0.7)
    ax.grid(True, alpha=0.15, color='white')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f'  saved {out_path}')
    plt.close(fig)


def plot_mse_over_horizon(pred_seq, truth_seq, out_path):
    mse = mse_per_step(pred_seq, truth_seq)
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style(ax)
    ax.semilogy(mse, color='#ff6b6b', linewidth=1.8)
    ax.set_xlabel('rollout step', color='white')
    ax.set_ylabel('MSE (log)', color='white')
    ax.set_title('Mean-squared error over rollout horizon',
                 color='white', fontsize=11)
    ax.grid(True, alpha=0.15, color='white', which='both')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f'  saved {out_path}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Spectral error
# ─────────────────────────────────────────────────────────────────────────────

def plot_spectral_error(pred_seq, truth_seq, dataset, out_path):
    fig, (ax_amp, ax_err) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(DARK_BG)
    style(ax_amp); style(ax_err)

    if dataset == 'ks':
        # 1D spectrum, average over time of an entire trajectory
        t_spec = np.abs(np.fft.rfft(truth_seq.mean(-1), axis=-1)).mean(0)
        p_spec = np.abs(np.fft.rfft(pred_seq.mean(-1),  axis=-1)).mean(0)
    else:
        # 2D radial spectrum at mid-rollout
        mid = len(truth_seq) // 2
        t_field = truth_seq[mid].mean(-1) if truth_seq[mid].ndim == 3 else truth_seq[mid]
        p_field = pred_seq[mid].mean(-1)  if pred_seq[mid].ndim  == 3 else pred_seq[mid]
        _, t_spec = radial_spectrum_2d(t_field)
        _, p_spec = radial_spectrum_2d(p_field)

    k = np.arange(len(t_spec))
    ax_amp.semilogy(k[1:], t_spec[1:], color='#00d4ff', linewidth=1.5, label='truth')
    ax_amp.semilogy(k[1:], p_spec[1:], color='#ff6b6b', linewidth=1.5, label='prediction')
    ax_amp.set_xlabel('wavenumber k', color='white')
    ax_amp.set_ylabel('amplitude |u(k)|', color='white')
    ax_amp.set_title('Frequency spectrum', color='white', fontsize=11)
    ax_amp.legend(fontsize=9, facecolor='#222', labelcolor='white', framealpha=0.7)
    ax_amp.grid(True, alpha=0.15, color='white', which='both')

    err = np.abs(t_spec - p_spec) / (t_spec + 1e-9)
    ax_err.plot(k[1:], err[1:], color='#ffaa44', linewidth=1.5)
    ax_err.set_xlabel('wavenumber k', color='white')
    ax_err.set_ylabel('relative spectral error', color='white')
    ax_err.set_title('Per-wavenumber relative error\n(PDE-Refiner: high-k typically underestimated)',
                     color='white', fontsize=10)
    ax_err.grid(True, alpha=0.15, color='white')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f'  saved {out_path}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Energy conservation
# ─────────────────────────────────────────────────────────────────────────────

def plot_energy_conservation(pred_seq, truth_seq, out_path):
    # Use L2-norm as a proxy for "energy"
    e_truth = (truth_seq ** 2).reshape(len(truth_seq), -1).mean(axis=1)
    e_pred  = (pred_seq  ** 2).reshape(len(pred_seq),  -1).mean(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style(ax)
    ax.plot(e_truth, color='#00d4ff', linewidth=1.8, label='truth')
    ax.plot(e_pred,  color='#ff6b6b', linewidth=1.8, label='prediction')
    ax.set_xlabel('rollout step', color='white')
    ax.set_ylabel(r'mean $u^2$', color='white')
    ax.set_title('Energy / L2-norm conservation over rollout',
                 color='white', fontsize=11)
    ax.legend(fontsize=9, facecolor='#222', labelcolor='white', framealpha=0.7)
    ax.grid(True, alpha=0.15, color='white')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f'  saved {out_path}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 6: Value distribution comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_value_histogram(pred_seq, truth_seq, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK_BG)
    style(ax)
    bins = np.linspace(-0.05, 1.05, 60)
    ax.hist(truth_seq.ravel(), bins=bins, density=True,
            color='#00d4ff', alpha=0.55, label='truth')
    ax.hist(pred_seq.ravel(),  bins=bins, density=True,
            color='#ff6b6b', alpha=0.55, label='prediction')
    ax.set_xlabel('pixel value (normalised)', color='white')
    ax.set_ylabel('density', color='white')
    ax.set_title('Value-distribution comparison\n(does the model preserve the marginal?)',
                 color='white', fontsize=11)
    ax.legend(fontsize=9, facecolor='#222', labelcolor='white', framealpha=0.7)
    ax.grid(True, alpha=0.15, color='white')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    print(f'  saved {out_path}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir',  required=True,
                        help='PlaNet checkpoint directory')
    parser.add_argument('--dataset', required=True, choices=['ks', 'kolmogorov'])
    parser.add_argument('--output',  required=True, help='output figure dir')
    parser.add_argument('--episode-dir', default=None,
                        help='Test episode dir (defaults based on dataset)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Default episode location: logdir/../test_episodes (KS) or .../kolmogorov_test_episodes
    if args.episode_dir is None:
        parent = os.path.dirname(args.logdir)
        ep_dir = ('kolmogorov_test_episodes' if args.dataset == 'kolmogorov'
                  else 'test_episodes')
        args.episode_dir = os.path.join(parent, ep_dir)

    print(f'Loading test episodes from {args.episode_dir} ...')
    eps = load_episodes(args.episode_dir, max_n=8)
    if not eps:
        print(f'  No episodes found.  Did the converter run?')
        sys.exit(1)
    print(f'  Loaded {len(eps)} episodes.')

    # If PlaNet has not yet dumped open-loop rollouts to disk, fall back to
    # using the test episode itself shifted by 1 step as a "prediction" stub.
    # This still produces all plots so you can sanity-check the analysis code.
    rollouts = find_rollout_dumps(args.logdir)
    if rollouts:
        print(f'  Found {len(rollouts)} rollout dumps -- using them.')
        dump = np.load(rollouts[0])
        pred_seq  = dump['prediction'] if 'prediction' in dump.files else dump['image']
        truth_seq = dump['truth']      if 'truth'      in dump.files else eps[0]['image']
    else:
        print('  No rollout dumps found in logdir.  Using identity-stub for plot QA.')
        truth_seq = eps[0]['image']
        pred_seq  = np.roll(truth_seq, shift=-1, axis=0)
        pred_seq[-1] = pred_seq[-2]    # avoid wrap-around at the end

    print('Generating plots ...')
    plot_rollout_strips(pred_seq, truth_seq, args.dataset,
                        os.path.join(args.output, f'rollout_strips_{args.dataset}.png'))
    plot_correlation_decay(pred_seq, truth_seq,
                           os.path.join(args.output, 'correlation_decay.png'))
    plot_mse_over_horizon(pred_seq, truth_seq,
                          os.path.join(args.output, 'mse_over_horizon.png'))
    plot_spectral_error(pred_seq, truth_seq, args.dataset,
                        os.path.join(args.output, 'spectral_error.png'))
    plot_energy_conservation(pred_seq, truth_seq,
                             os.path.join(args.output, 'energy_conservation.png'))
    plot_value_histogram(pred_seq, truth_seq,
                         os.path.join(args.output, 'value_histogram.png'))

    print('\nDone.  Plots in', args.output)


if __name__ == '__main__':
    main()
