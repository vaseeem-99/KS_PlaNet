"""
visualize_both.py
=================
Place at: planet/ks_scripts/visualize_both.py

Side-by-side comparison of KS (1D) and Kolmogorov (2D) datasets.
Generates one figure showing:
  - example trajectories from each dataset
  - frequency-spectrum comparison
  - sample-count and resolution summary

Run from planet/:
    python ks_scripts/visualize_both.py

Outputs:
    both_datasets_comparison.png
"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ks_dataset')
SAMPLE_KS = 0
SAMPLE_KO = 0
CMAP      = 'RdBu_r'
# ─────────────────────────────────────────────────────────────────────────────


def radial_spectrum(field_2d):
    """2D FFT -> 1D radial spectrum."""
    h, w  = field_2d.shape
    fft2  = np.fft.fftshift(np.fft.fft2(field_2d))
    power = np.abs(fft2) / (h * w)
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r    = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    k_max = min(cy, cx)
    radial = np.zeros(k_max)
    for k in range(k_max):
        mask = (r == k)
        if mask.any():
            radial[k] = power[mask].mean()
    return np.arange(k_max), radial


def style(ax):
    ax.set_facecolor('#111')
    ax.tick_params(colors='white', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')


# ─────────────────────────────────────────────────────────────────────────────
# Load both datasets (memory-mapped)
# ─────────────────────────────────────────────────────────────────────────────

print('Loading KS dataset...')
ks_train_path = os.path.join(DATA_DIR, 'ks_train.npz')
ks_data       = np.load(ks_train_path, mmap_mode='r')
u_ks          = ks_data['u']                     # (4096, 140, 256)
print(f'  KS  u: {u_ks.shape}')

print('Loading Kolmogorov dataset...')
ko_X_path = os.path.join(DATA_DIR, 'X_train.npy')
ko_Y_path = os.path.join(DATA_DIR, 'Y_train.npy')
X_ko      = np.load(ko_X_path, mmap_mode='r')    # (140000, 1, 128, 128)
Y_ko      = np.load(ko_Y_path, mmap_mode='r')
print(f'  Kolmogorov X: {X_ko.shape}  Y: {Y_ko.shape}')


# ─────────────────────────────────────────────────────────────────────────────
# Sequential vs independent quick check on Kolmogorov
# ─────────────────────────────────────────────────────────────────────────────
def quick_seq_check(X, Y, n=10):
    idxs = np.linspace(0, len(X) - 2, n).astype(int)
    diffs = []
    for i in idxs:
        d = np.abs(Y[i].ravel() - X[i + 1].ravel()).mean()
        denom = np.abs(Y[i]).mean() + 1e-6
        diffs.append(d / denom)
    return float(np.mean(diffs))

seq_diff = quick_seq_check(X_ko, Y_ko)
is_seq   = seq_diff < 1e-3
print(f'\nKolmogorov pair check: Y[i] vs X[i+1] mean rel diff = {seq_diff:.4f}')
print(f'  layout: {"SEQUENTIAL" if is_seq else "INDEPENDENT PAIRS"}')


# ─────────────────────────────────────────────────────────────────────────────
# Build comparison figure
# ─────────────────────────────────────────────────────────────────────────────
print('\nBuilding comparison figure...')

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0d0d0d')
gs  = GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.3,
               height_ratios=[1, 1, 1])

# ─── Row 1: KS space-time heatmap (1D x time) ────────────────────────────────
ax_ks = fig.add_subplot(gs[0, :])
style(ax_ks)
ks_traj = u_ks[SAMPLE_KS]                         # (140, 256)
L_val   = float(ks_data['L'][SAMPLE_KS])
dt_val  = float(ks_data['dt'][SAMPLE_KS])
ax_ks.imshow(ks_traj, aspect='auto', origin='lower',
             extent=[0, L_val, 0, ks_traj.shape[0] * dt_val],
             cmap=CMAP, vmin=-3.7, vmax=3.7, interpolation='bilinear')
ax_ks.set_xlabel('space x', color='white', fontsize=10)
ax_ks.set_ylabel('time t (s)', color='white', fontsize=10)
ax_ks.set_title(
    f'KS (1D)  --  Trajectory {SAMPLE_KS}  |  spatial: 256 points  |  '
    f'temporal: 140 steps  |  domain L={L_val:.1f}',
    color='white', fontsize=11)


# ─── Row 2: Kolmogorov frame pair (X input, Y target) ────────────────────────
ax_ko_X = fig.add_subplot(gs[1, 0])
ax_ko_Y = fig.add_subplot(gs[1, 1])
ax_ko_d = fig.add_subplot(gs[1, 2])    # difference Y - X
style(ax_ko_X); style(ax_ko_Y); style(ax_ko_d)

x_field = np.asarray(X_ko[SAMPLE_KO])[0]
y_field = np.asarray(Y_ko[SAMPLE_KO])[0]
diff    = y_field - x_field

ax_ko_X.imshow(x_field, cmap=CMAP, vmin=-7.5, vmax=7.5, origin='lower')
ax_ko_X.set_title(f'Kolmogorov INPUT X[{SAMPLE_KO}]',
                  color='#00d4ff', fontsize=10)
ax_ko_X.set_xticks([]); ax_ko_X.set_yticks([])

ax_ko_Y.imshow(y_field, cmap=CMAP, vmin=-7.5, vmax=7.5, origin='lower')
ax_ko_Y.set_title(f'Kolmogorov TARGET Y[{SAMPLE_KO}]',
                  color='#ff6b6b', fontsize=10)
ax_ko_Y.set_xticks([]); ax_ko_Y.set_yticks([])

ax_ko_d.imshow(diff, cmap=CMAP, vmin=-2, vmax=2, origin='lower')
ax_ko_d.set_title('Y - X  (one-step change)',
                  color='white', fontsize=10)
ax_ko_d.set_xticks([]); ax_ko_d.set_yticks([])

# stats text panel
ax_stats = fig.add_subplot(gs[1, 3])
ax_stats.axis('off')
stats_text = (
    f'DATASET COMPARISON\n\n'
    f'KS (1D):\n'
    f'  shape    : {u_ks.shape}\n'
    f'  spatial  : 256 -> 64 (subsampled)\n'
    f'  temporal : 140 steps per traj\n'
    f'  range    : [-3.7, 3.7]\n'
    f'  n_traj   : 4096 (train)\n\n'
    f'Kolmogorov (2D):\n'
    f'  shape    : {X_ko.shape}\n'
    f'  spatial  : 128x128 -> 64x64\n'
    f'  temporal : 2 frames per pair\n'
    f'  range    : [-7.5, 7.5]\n'
    f'  n_pairs  : 140000 (train)\n'
    f'  layout   : {"sequential" if is_seq else "independent pairs"}\n'
    f'  pair_diff: {seq_diff:.4f}\n'
)
ax_stats.text(0.02, 0.98, stats_text, transform=ax_stats.transAxes,
              color='white', fontsize=9, va='top', family='monospace')


# ─── Row 3: Frequency spectra comparison ─────────────────────────────────────
ax_spec = fig.add_subplot(gs[2, :2])
style(ax_spec)

# KS spectrum: 1D FFT along spatial axis, average over time
ks_spec = np.abs(np.fft.rfft(ks_traj, axis=1)).mean(axis=0) / 256
ks_k    = np.arange(len(ks_spec))
ax_spec.semilogy(ks_k[1:], ks_spec[1:], color='#00d4ff', linewidth=1.5,
                 label='KS  (1D spatial spectrum)')

# Kolmogorov spectrum: 2D radial average
ko_k, ko_spec = radial_spectrum(x_field)
ax_spec.semilogy(ko_k[1:], ko_spec[1:], color='#ff6b6b', linewidth=1.5,
                 label='Kolmogorov  (2D radial spectrum)')

ax_spec.set_xlabel('wavenumber k', color='white', fontsize=10)
ax_spec.set_ylabel('amplitude |u(k)|', color='white', fontsize=10)
ax_spec.set_title('Frequency Spectrum Comparison (log scale)',
                  color='white', fontsize=11)
ax_spec.legend(fontsize=9, facecolor='#222', labelcolor='white',
               loc='lower left', framealpha=0.7)
ax_spec.grid(True, alpha=0.15, color='white')


# ─── Row 3 (right): PlaNet training implications ─────────────────────────────
ax_impl = fig.add_subplot(gs[2, 2:])
ax_impl.axis('off')
impl_text = (
    'PLANET TRAINING IMPLICATIONS\n\n'
    'KS (full trajectories):\n'
    '  batch_shape = (50, 40)\n'
    '  chunks per traj = floor(140/40) = 3\n'
    '  total chunks = 4096 x 3 = 12,288\n'
    '  RSSM learns LONG-HORIZON dynamics\n'
    '  -> standard PlaNet setup\n\n'
    'Kolmogorov (independent pairs):\n'
    '  batch_shape = (64, 2)\n'
    '  no long-horizon BPTT possible\n'
    '  RSSM learns ONE-STEP dynamics\n'
    '  -> valid latent world model still\n'
    '  -> faster training per epoch\n'
    '  -> compare to LNS/PDE-Refiner baselines\n'
)
ax_impl.text(0.02, 0.98, impl_text, transform=ax_impl.transAxes,
             color='white', fontsize=9, va='top', family='monospace')

fig.suptitle(
    'KS (1D) vs Kolmogorov (2D)  --  Both Datasets Through the PlaNet Pipeline',
    color='white', fontsize=13, y=0.995)

out_path = os.path.join(os.path.dirname(__file__), 'both_datasets_comparison.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d0d0d')
print(f'  Saved: {out_path}')
plt.show()

print('\nDone.')
