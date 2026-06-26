# PLOTS_GUIDE — Every plot this pipeline produces

Three categories, in the order they appear in your workflow:

| Phase | Source | Purpose |
|---|---|---|
| A. Data validation (pre-training) | `ks_scripts/visualize_*.py` | Sanity-check raw + converted data |
| B. Training-time monitoring | W&B (auto-logged) | Watch convergence and detect failures live |
| C. Post-training research plots | `analyze_results.py` | The figures you put in a paper |

---

## A. Data validation plots (run before any training)

These come from your existing `visualize_ks.py`, the `visualize_kolmogorov.py`
I wrote for you, and `visualize_both.py`. Run them once after a converter
finishes; they prove the data is shaped and rendered correctly.

### A1. `ks_heatmap.png`
- **What**: Space-time heatmap of one KS trajectory. X-axis = space, Y-axis = time, color = `u(x,t)` via `RdBu_r`.
- **Good**: Sloped chevrons fanning out into chaos after the warmup. Symmetric color distribution around zero.
- **Bad**: All one color (normalization off), or vertical stripes (time/space axes swapped).
- **Why it matters**: KS is the canonical 1D spatiotemporal chaos benchmark. The chevron pattern is the signature of mode coupling — if it's missing, you're not actually looking at KS.

### A2. `ks_spectrum.png`
- **What**: Two panels — waveforms at several timesteps (left), and the spectral amplitude `|û(k)|` on log scale (right) with the high-`k` band shaded red.
- **Good**: Spectrum that decays approximately exponentially in `k`, with most energy in the first 20-30 modes and ~6 orders of magnitude drop by `k=128`.
- **Why it matters**: PDE-Refiner's central claim is that MSE-trained models systematically underestimate the low-amplitude high-`k` modes. This plot establishes what the "true" spectrum looks like so you can later compare model output against it.

### A3. `ks_multi_traj.png`
- **What**: 8 different trajectories side by side as heatmaps, each with its own domain length `L`.
- **Why it matters**: Confirms the dataset has genuine diversity. If trajectories look identical, the data loader is broken.

### A4. `ks_animation.gif`
- **What**: One trajectory animated over time — waveform, space-time cursor, and spectrum all updating together.
- **Use**: Sanity-check + slide-deck material for presenting the dataset.

### A5. `kolmogorov_snapshots.png`
- **What**: One Kolmogorov sample shown as a 2D heatmap. With your data this is just a single frame because each sample is one snapshot, not a trajectory.
- **Good**: Visible vortices/turbulent structure with red/blue alternating regions.
- **Bad**: Smooth/laminar field (Reynolds too low), or pure noise (data corrupted in conversion).

### A6. `kolmogorov_xy_pair.png`  ← most important for understanding the task
- **What**: Input frame `X[i]` (top), target frame `Y[i]` (bottom), and difference `Y - X` (right). Side panel shows dataset statistics.
- **Good**: `X` and `Y` look similar but with small visible structural change. Diff plot has visible structure (not constant).
- **Why it matters**: Tells you the dataset's effective Δt at a glance. If `Y` is identical to `X`, your Δt is too small to be interesting. If `Y` looks unrelated to `X`, Δt is too large for learning.

### A7. `kolmogorov_diversity.png`
- **What**: 8 different `X` samples at the same conceptual timestep.
- **Why it matters**: Confirms the dataset spans diverse initial conditions, not just rotations of the same vortex.

### A8. `kolmogorov_spectrum.png`
- **What**: 2D radial average of the FFT — equivalent to A2 but isotropized for the 2D field.
- **Good**: Power-law decay characteristic of forced 2D turbulence. The Kraichnan-style `k^{-5/3}` or `k^{-3}` inertial range may be visible at intermediate `k`.
- **Why it matters**: Forced 2D Kolmogorov flow has a known statistical signature. Recovering this in your data validates the source simulator.

### A9. `kolmogorov_animation.gif`
- **What**: One sample over time (with three panels updating). Only meaningful if the converter detected sequential data.

### A10. `both_datasets_comparison.png`
- **What**: One figure showing KS space-time slice (top), Kolmogorov input/target/diff (middle), and overlaid frequency spectra of both datasets (bottom-left) + training-implications panel (bottom-right).
- **Use**: First figure of your paper. Shows the contrast between 1D long-trajectory chaos and 2D pair-based prediction at a glance.

---

## B. Training-time monitoring (W&B, auto-logged)

Open `wandb.ai/Physics_Sim/KS_PlaNet`, filter by tag `training`. The
following appear under both `ks_planet_ks_run` and `ks_planet_kolmogorov_run`.

### B1. `train/objectives_image`
- **What**: Image reconstruction log-likelihood per pixel.
- **Healthy**: Rises monotonically toward zero. Plateaus eventually.
- **Failure mode**: Stays flat at a large negative value → encoder/decoder mismatch, batch shape wrong, or NaN.

### B2. `train/objectives_divergence`
- **What**: KL(posterior ‖ prior) of the latent state, the regulariser term.
- **Healthy**: Rises briefly during the first 5–10k steps as the posterior gets informative, then falls and stabilises at 2–10 nats.
- **Failure mode**: Stays at zero → posterior collapsed to the prior, model is useless. Reduce `free_nats` from 2.0 to 1.0.

### B3. `train/closedloop_state_posterior_entropy` vs `train/closedloop_state_prior_entropy`
- **What**: Differential entropy of the two latent distributions.
- **Healthy**: Posterior < Prior. The encoder is making the distribution narrower (more certain) given the observation.
- **Failure mode**: Posterior ≥ Prior. The encoder is ignoring its input.

### B4. `train/dist_summaries_image_abs_error`
- **What**: Mean absolute pixel error between reconstructed and ground-truth images.
- **Healthy**: Falls below 0.05 (on the [0, 1] normalised pixel scale).

### B5. `test/mean_score`
- **What**: Open-loop reconstruction score on held-out episodes.
- **Healthy**: Rises noisily through training. Tracks training loss with some gap.
- **Failure mode**: Falls or stays at zero → catastrophic overfitting.

### B6. `train/general_steps_per_second`
- **What**: Throughput.
- **Healthy on A100**: ~3-5 steps/s for KS at `batch_shape=(50, 40)`, ~6-10 for Kolmogorov at `(64, 2)`.

### B7. W&B image panels
- `closedloop/posterior/image` — reconstruction WITH access to all frames. Should be near-perfect.
- `closedloop/prior/image` — reconstruction WITHOUT access to future frames, just from prior dynamics. The interesting one — this is the world model's actual capability.
- `openloop/image` — long rollout from an initial latent state. The "how-far-can-you-predict" view.

Each panel is a strip: top row = truth, bottom row = model output. Compare visually.

---

## C. Post-training research plots (`analyze_results.py`)

Run after training. These are what you put in a paper.

```bash
python analyze_results.py \
    --logdir checkpoints/ks_run \
    --dataset ks \
    --output figures_ks

python analyze_results.py \
    --logdir checkpoints/kolmogorov_run \
    --dataset kolmogorov \
    --output figures_kolmogorov
```

### C1. `rollout_strips_{ks,kolmogorov}.png`
- **What**: Two-row strip showing ground truth (top) and predicted (bottom) at rollout horizons 0, 5, 10, 20, 50 steps.
- **Use**: First impression of model quality. If predictions diverge visibly by step 10, your effective prediction horizon is short.

### C2. `correlation_decay.png`
- **What**: Pearson correlation between predicted and true fields, plotted against rollout step. Marks where `r = 0.9` and `r = 0.8` are crossed.
- **Why it matters**: **This is the standard PDE-solver benchmark metric.** PDE-Refiner reports KS predictions stable up to step ~196 (98 seconds at dt=0.5) by this metric. Your "high correlation time" goes in your results table.

### C3. `mse_over_horizon.png`
- **What**: Per-step MSE on log scale.
- **Healthy**: Roughly exponential growth of error with rollout step (characteristic of chaotic systems hitting their Lyapunov time).
- **Failure mode**: Flat at maximum → model produces constant output (collapse).

### C4. `spectral_error.png`  ← most important diagnostic for PDE work
- **What**: Two panels. Left: predicted vs true frequency spectrum overlaid (log scale). Right: relative spectral error per wavenumber.
- **Why it matters**: This is the plot that motivates everything in PDE-Refiner. If the model's high-`k` error is much larger than its low-`k` error, MSE training has indeed neglected the small-amplitude modes. Latent-space approaches like yours might do better, *or* might also fail — this plot tells you which.

### C5. `energy_conservation.png`
- **What**: Mean `u²` (proxy for kinetic energy or KS L2-invariant) plotted over rollout time, truth vs prediction.
- **Why it matters**: Physical PDEs conserve or have constrained energy. A neural model that drifts the energy is unphysical. This plot diagnoses whether your latent dynamics preserve the physics.

### C6. `value_histogram.png`
- **What**: Marginal value distribution — histogram of all pixel values from truth and prediction.
- **Why it matters**: A model can have low MSE while still producing a wrong marginal distribution (e.g., over-smoothing means the histogram is too narrow). This catches that.

---

## What plots you'll generate and when

| Phase | Plots | Time |
|---|---|---|
| After Phase 4a (conversion) | A1–A10 (run visualizers locally on a sample) | 1 min each |
| During Phase 4b (training) | B1–B7 (live in W&B) | continuous |
| After Phase 5 (download) | C1–C6 for each dataset | ~30 s each |

For a paper / thesis you usually want:
- **A10** as a "dataset" figure
- **B7** image panels as a qualitative-results figure
- **C2** correlation decay as the headline quantitative result
- **C4** spectral error as the diagnostic discussion figure

---

## Why these and not others?

This set is curated for **PDE-solver / physics-simulation / latent-dynamics**
research specifically. It mirrors what PDE-Refiner (Lippe et al. 2023), LNS
(Li et al. 2025), and PNO (Bülte et al. 2025) report. If you compare your
results to those baselines, the metrics will be directly comparable.

What's deliberately NOT here:
- Latent traversals / interpolation — interesting for VAE work but not standard PDE benchmarks
- Lyapunov exponent estimation — requires gradient analysis of the learned dynamics, harder to do well
- Conservation-law verification beyond energy (e.g., momentum, enstrophy) — would need physics-specific code; the L2-norm plot is a good proxy
