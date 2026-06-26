# KS-PlaNet: World Model Learning for Chaotic PDE Dynamics

> **PlaNet** (Deep Planning Network) adapted to learn latent dynamics models of two chaotic PDE systems:
> - **Kuramoto-Sivashinsky (KS)** — 1D fourth-order chaos PDE
> - **Kolmogorov flow** — 2D incompressible Navier-Stokes variant
>
> Both pipelines write to the same W&B project (`Physics_Sim/KS_PlaNet`), distinguished by tags.

---

## What This Project Does

This project trains PlaNet's RSSM (Recurrent State Space Model) on two fundamentally different PDE benchmarks and compares latent-space dynamics learning across them. The two systems are deliberately chosen for contrast: KS is one-dimensional in space but exhibits temporal chaos over long trajectories; Kolmogorov flow is two-dimensional but the dataset is structured as one-step prediction pairs.

This contrast lets you investigate whether PlaNet's strength (long-horizon BPTT through a latent dynamics model) generalises from a 1D long-trajectory setting to a 2D short-horizon setting.

---

## Dataset Comparison

| Property | KS (1D) | Kolmogorov (2D) |
|----------|---------|-----------------|
| Native resolution | 256 spatial points | 128 × 128 grid |
| Rendered for PlaNet | 64×64×3 | 64×64×3 |
| Native value range | [-3.7, 3.7] | [-7.5, 7.5] |
| Train samples | 4,096 trajectories × 140 steps | 140,000 (X, Y) pairs |
| Test samples | 512 trajectories × 640 steps | 30,000 (X, Y) pairs |
| Val samples | — | 30,000 (X, Y) pairs |
| Episode structure | Long trajectories | Independent pairs (or rebuilt trajectories) |
| PlaNet chunk size | L = 40 | L = 2 (or L = 51 if sequential) |
| Source files | `ks_train.npz`, `ks_test.npz` | `X_*.npy`, `Y_*.npy` |

The Kolmogorov data is shaped `(N, 1, 128, 128)` — channel-first single-channel (vorticity). The "1" axis is channels, not time. Each `X[i] → Y[i]` is one supervised prediction pair at some Δt.

---

## Codebase Structure

```
planet/
├── setup.py
├── requirements.txt
├── README.md                              ← this file
│
├── ks_dataset/                            ← all source + converted data
│   ├── ks_train.npz, ks_test.npz          KS source
│   ├── X_train.npy, X_test.npy, X_val.npy
│   ├── Y_train.npy, Y_test.npy, Y_val.npy Kolmogorov source
│   ├── train_episodes/                    KS converted (4096 NPZ)
│   ├── test_episodes/                     KS converted (512 NPZ)
│   ├── kolmogorov_train_episodes/         Kolmogorov converted
│   ├── kolmogorov_test_episodes/
│   └── kolmogorov_val_episodes/
│
├── ks_scripts/                            ← conversion + visualization
│   ├── __init__.py
│   ├── convert_ks_to_planet.py            STEP 1a: prepare KS data
│   ├── convert_kolmogorov_to_planet.py    STEP 1b: prepare Kolmogorov data
│   ├── visualize_ks.py                    (existing) inspect KS data
│   ├── visualize_kolmogorov.py            inspect Kolmogorov data
│   └── visualize_both.py                  side-by-side comparison
│
├── scripts/
│   ├── train.py                           unchanged
│   ├── configs.py                         + ks_config + kolmogorov_config at end
│   ├── tasks.py                           + ks_chaos + kolmogorov_flow at end
│   └── objectives.py                      unchanged
│
├── tools/
│   ├── numpy_episodes.py                  + ks_episode_reader at end (used for both)
│   └── ...                                everything else unchanged
│
├── training/
│   ├── wandb_logger.py                    central W&B logging
│   ├── trainer.py                         W&B hook in iterate()
│   ├── utility.py                         W&B init/finish in train()
│   └── ...
│
├── models/
│   └── rssm.py                            RSSM cell (unchanged)
│
└── networks/
    └── conv_ha.py                         Ha-Schmidhuber CNN (unchanged)
```

### Files edited vs files untouched

| File | Status | Change |
|------|--------|--------|
| `ks_scripts/convert_ks_to_planet.py` | **UPDATED** | Now in `ks_scripts/`, path goes up to `../ks_dataset/` |
| `ks_scripts/convert_kolmogorov_to_planet.py` | **NEW** | Auto-detects sequential vs pair layout |
| `ks_scripts/visualize_both.py` | **NEW** | Side-by-side dataset comparison |
| `tools/numpy_episodes.py` | **APPEND** | `ks_episode_reader` — same reader for both datasets |
| `scripts/tasks.py` | **APPEND** | `ks_chaos` + `kolmogorov_flow` tasks |
| `scripts/configs.py` | **APPEND** | `ks_config` + `kolmogorov_config` |
| `training/` (all files) | **unchanged** | W&B integration was already done |
| `models/`, `networks/`, `control/` | **unchanged** | RSSM and CNN backbone same for both |

---

## Installation

```bash
git clone https://github.com/google-research/planet.git
cd planet

conda create -n ksplanet python=3.7
conda activate ksplanet

pip install -r requirements.txt
wandb login   # paste API key from https://wandb.ai/authorize
```

GPU users: keep `tensorflow-gpu==1.13.1` in `requirements.txt`. CPU-only: swap for `tensorflow==1.13.1`.

---

## Step 1 — Convert Data

### KS

```bash
python ks_scripts/convert_ks_to_planet.py
```

Renders each 1D `u(x, t)` row to a 64×64×3 image via the `RdBu_r` colormap. Produces 4,608 NPZ files (~11 GB).

### Kolmogorov

```bash
python ks_scripts/convert_kolmogorov_to_planet.py
```

At startup the converter checks whether `Y[i] == X[i+1]` across a sample of indices and prints which layout it detected:

```
Checking whether data is sequential (Y[i] == X[i+1])...
  Mean relative diff between Y[i] and X[i+1]: 0.4123
  Layout: INDEPENDENT PAIRS
```

By default it converts a subset (20k train, 5k test, 5k val pairs) — these defaults sit at the top of the script and you can change them to `None` for the full 200k. Each pair becomes a 2-frame mini-episode; if data turns out to be sequential, trajectories of length 51 are built instead.

**Both conversions log live to W&B:**
- Failure tables (`train_failures`, `test_failures`, `val_failures`) — capture full traceback per failed episode
- Progress scalars (`*/episodes_done`, `*/conversion_rate`)
- Final disk stats and `conversion_status: PASSED / FAILED`
- W&B alert on any failure

---

## Step 2 — Visualise

Run any of these from `planet/`:

```bash
python ks_scripts/visualize_ks.py            # KS heatmap + spectrum + animation
python ks_scripts/visualize_kolmogorov.py    # Kolmogorov snapshots + X→Y + spectrum + animation
python ks_scripts/visualize_both.py          # side-by-side comparison
```

The third script is the most useful for understanding the contrast — it shows a KS space-time slice next to a Kolmogorov input/target pair, with both frequency spectra overlaid on the same axes.

---

## Step 3 — Train PlaNet

### KS

```bash
python -m planet.scripts.train \
    --logdir ks_dataset/ks_run \
    --config ks_config \
    --params "{tasks: [ks_chaos]}"
```

Reads from `ks_dataset/train_episodes/` and `ks_dataset/test_episodes/`. Writes checkpoints to `ks_dataset/ks_run/`. W&B run name: `ks_planet_ks_run`.

### Kolmogorov

```bash
python -m planet.scripts.train \
    --logdir ks_dataset/kolmogorov_run \
    --config kolmogorov_config \
    --params "{tasks: [kolmogorov_flow]}"
```

Reads from `ks_dataset/kolmogorov_train_episodes/` and `ks_dataset/kolmogorov_test_episodes/`. Writes to `ks_dataset/kolmogorov_run/`. W&B run name: `ks_planet_kolmogorov_run`.

If the converter detected sequential data and rebuilt trajectories of length 51, override the chunk shape:

```bash
python -m planet.scripts.train \
    --logdir ks_dataset/kolmogorov_run \
    --config kolmogorov_config \
    --params "{tasks: [kolmogorov_flow], batch_shape: [50, 40], max_length: 51, open_loop_context: 20}"
```

### Hyperparameter differences

| Parameter | `ks_config` | `kolmogorov_config` |
|-----------|-------------|---------------------|
| `batch_shape` | (50, 40) | (64, 2) |
| `open_loop_context` | 20 | 1 |
| `model_size` | 400 | 400 |
| `state_size` | 64 | 64 |
| `free_nats` | 2.0 | 2.0 |
| `image_bits` | 8 | 8 |
| Episode source | `train_episodes/` | `kolmogorov_train_episodes/` |
| Effective task | long-horizon BPTT | one-step latent dynamics |

---

## Step 4 — Monitor in W&B

All runs go to **[wandb.ai/Physics_Sim/KS_PlaNet](https://wandb.ai/Physics_Sim/KS_PlaNet)**.

Filter by tag to separate the two experiments:
- Tag `ks` — KS conversion + training
- Tag `kolmogorov` — Kolmogorov conversion + training
- Tag `data_conversion` — both converters
- Tag `training` — both training runs

### Run names

| Run | Tag | Source |
|-----|-----|--------|
| `ks_data_conversion` | ks | KS converter |
| `kolmogorov_data_conversion` | kolmogorov | Kolmogorov converter |
| `ks_planet_ks_run` | ks | KS training |
| `ks_planet_kolmogorov_run` | kolmogorov | Kolmogorov training |

### Scalars to watch

For both training runs the W&B logger parses TF Summary protobuf bytes and extracts every scalar tag. The interesting ones:

- `train/objectives_image` — image reconstruction log-likelihood (rising)
- `train/objectives_divergence` — KL(posterior ∥ prior) (falls then stabilises)
- `train/closedloop_state_posterior_entropy` — should be lower than prior entropy
- `train/closedloop_state_prior_entropy`
- `train/dist_summaries_image_abs_error` — mean pixel error (falling)
- `test/mean_score` — open-loop prediction score
- `train/general_steps_per_second` — throughput

---

## How the Data Flows

**KS path:**
```
ks_train.npz (4096, 140, 256)
        │
        ▼  convert_ks_to_planet.py
train_episodes/  4096 × episode_NNNNN.npz  shape: image=(140, 64, 64, 3)
        │
        ▼  ks_episode_reader → cache_loader → chunk_sequence(L=40)
batches of (B=50, L=40, 64, 64, 3)
        │
        ▼  conv_ha encoder → RSSM (state=64, hidden=400) → decoder
ELBO loss: image log-prob + KL(posterior‖prior, free_nats=2)
        │
        ▼  Adam(lr=1e-3, ε=1e-4, grad_clip=1000)
W&B scalars
```

**Kolmogorov path:**
```
X_train.npy (140000, 1, 128, 128)        Y_train.npy (140000, 1, 128, 128)
                    │
                    ▼  detect_sequential() → INDEPENDENT or SEQUENTIAL
        (independent)│                          (sequential)│
                    ▼                                       ▼
        kolmogorov_train_episodes/         kolmogorov_train_episodes/
        20000 × episode_NNNNNN.npz         400 × episode_NNNNNN.npz
        shape: image=(2, 64, 64, 3)        shape: image=(51, 64, 64, 3)
                    │                                       │
                    ▼  ks_episode_reader → cache_loader     ▼
        batches of (B=64, L=2, 64, 64, 3)  batches of (B=50, L=40, 64, 64, 3)
                    │                                       │
                    └────────────────┬──────────────────────┘
                                     ▼
                  conv_ha → RSSM → decoder → ELBO → Adam → W&B
```

Both paths share the same RSSM cell, encoder, decoder, optimizer, and W&B logging. Only the chunk shape, episode-source directory, and number of total chunks differ.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**
→ Python 3.8+. TF 1.13 requires Python 3.6 or 3.7.

**Converter crashes with `OSError: [Errno 12] Cannot allocate memory`**
→ Reduce `workers` at the top of the converter script.

**Kolmogorov training reports `InvalidArgumentError: batch_shape mismatch`**
→ You used `ks_config` defaults on Kolmogorov data, or vice versa. Match the chunk length to your episode length: 2 for pairs, 51 for rebuilt trajectories, 40 for KS.

**W&B says `wandb: ERROR Run initialization has timed out`**
→ Confirm `wandb login` was run, or set `WANDB_MODE=offline` to log locally.

**Kolmogorov spectrum looks identical to KS**
→ Make sure you ran `visualize_both.py` (not `visualize_ks.py`) and that both source files load correctly.

---

## Research Direction

The two datasets together let you investigate three research questions:

1. **Does latent-space dynamics learning generalise across spatial dimensionality?** PlaNet was designed for 64×64 pixel observations. Both KS-rendered and Kolmogorov-rendered fields share that shape, so the encoder/decoder/RSSM are identical; only the underlying physics differs.

2. **How does episode-length affect RSSM training?** KS uses L=40 chunks; Kolmogorov-pairs use L=2. Comparing training curves and final test prediction quality directly tests whether long BPTT is essential.

3. **Comparison to non-latent baselines.** PDE-Refiner reports 98 seconds of stable KS rollout at the same dataset configuration; LNS reports specific accuracy numbers on Kolmogorov at the same 64×64 setup. Both are public benchmarks for your latent approach to beat.

---

## References

- Hafner et al. (2019). *Learning Latent Dynamics for Planning from Pixels*. ICML. [arXiv:1811.04551](https://arxiv.org/abs/1811.04551)
- Lippe et al. (2023). *PDE-Refiner*. NeurIPS. [arXiv:2308.05732](https://arxiv.org/abs/2308.05732)
- Li et al. (2025). *Latent Neural PDE Solver*. JCP 524:113705.
- Bülte et al. (2025). *Probabilistic Neural Operators*. arXiv:2502.12902.
- Original PlaNet code: https://github.com/google-research/planet

---

## License

Apache 2.0. Original PlaNet © 2019 The PlaNet Authors. KS + Kolmogorov adaptation © 2026.
=======
