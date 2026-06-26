# KS-PlaNet Execution Guide

This document describes what happens at each stage of the pipeline, what
healthy output looks like, what plots to expect in W&B, and how to spot
problems early. Read this **before** submitting cluster jobs.

---

## Overall pipeline

```
                    SETUP (once)
                         │
                         ▼
            ┌──────────────────────┐
            │ 00_setup_polaris.sh  │   build container, install deps, login W&B
            └──────────────────────┘
                         │
   ┌─────────────────────┴─────────────────────┐
   │                                           │
   ▼                                           ▼
LOCAL                                       CLUSTER
dry_run/local_pipeline.sh                   qsub 01_convert.pbs
(verify on subset)                                 │
                                                   ▼
                                            qsub 02_train_ks.pbs
                                            qsub 03_train_kolmogorov.pbs
```

---

## Stage 1 — Local Dry-Run (WSL Ubuntu, ~10–15 min)

**Purpose:** verify every script runs end-to-end on a 50-trajectory KS subset
+ 200-pair Kolmogorov subset before committing cluster time.

```bash
cd planet/
bash dry_run/local_pipeline.sh
```

**TF 1.13 cannot use your RTX 3050.** The script forces CPU mode
(`CUDA_VISIBLE_DEVICES=""`) so the dry-run is portable. 200 training steps
on CPU finishes in ~2 minutes.

### What healthy console output looks like

```
[0/5] Source files present.
[1/5] Creating subset datasets ...
  -> ks_dataset/ks_train_subset.npz  (X.X MB)
  ...
[2/5] Running KS conversion on subset ...
  [train] 50/50 (100%)  50 ok  0 failed
[3/5] Running Kolmogorov conversion on subset ...
  Layout: INDEPENDENT PAIRS         <-- or SEQUENTIAL, both are fine
  [train] 200/200 (100%)  200 ok  0 failed
[4/5] Generating comparison visualisation ...
  Saved: both_datasets_comparison.png
[5/5] Running 200-step KS training to verify pipeline ...
  ... Epoch 1 phase train ...
  Score 0.234.
  Score 0.451.
...
Local dry-run PASSED.
```

### Failures and what they mean

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: tensorflow` | Wrong Python env | `conda activate ksplanet` |
| `np.bool deprecated` errors | numpy ≥ 1.20 | `pip install 'numpy<1.20'` |
| `ArrowInvalid` mid-conversion | Disk full / NPZ corruption | `df -h .`, re-download source |
| Training crashes on first batch | `ks_episode_reader` not appended | Append `tools_numpy_episodes_APPEND.py` content |
| Visualizer can't find file | Subset swap incomplete | Re-run step 1 of local_pipeline.sh |
| `cuDNN error: CUDNN_STATUS_NOT_INITIALIZED` | TF tried to use GPU | Make sure `CUDA_VISIBLE_DEVICES=""` is set |

When all 5 steps print `PASSED`, restore the originals:

```bash
cd ks_dataset && for f in *.full; do mv "$f" "${f%.full}"; done
```

---

## Stage 2 — Polaris Setup (login node, ~20 min)

```bash
ssh <user>@polaris.alcf.anl.gov
cd /eagle/projects/PIGAN/$USER/
mkdir -p ks_planet && cd ks_planet
# upload 00_setup_polaris.sh, then:
bash 00_setup_polaris.sh
```

This script:
1. Pulls the NGC TF1 container (~6 GB, one-time)
2. Creates the Apptainer overlay and installs PlaNet's pip extras into it
3. Prompts for the W&B API key

After it finishes, the cluster has:
```
/eagle/projects/PIGAN/<user>/ks_planet/
├── code/             (you copy planet/ here next)
├── data/             (unused -- data lives in code/planet/ks_dataset/)
├── containers/
│   ├── tf1_a100.sif        (~6 GB, the container)
│   └── overlay.img         (~2 GB, writable pip-install layer)
├── logs/             (job stdout/stderr land here)
└── checkpoints/      (PlaNet writes here)
```

Then upload the code:
```bash
# from your local machine
scp -r planet/ <user>@polaris.alcf.anl.gov:/eagle/projects/PIGAN/<user>/ks_planet/code/
```

And the data:
```bash
# raw source files into ks_dataset/
scp planet/ks_dataset/ks_train.npz planet/ks_dataset/ks_test.npz \
    planet/ks_dataset/X_*.npy planet/ks_dataset/Y_*.npy \
    <user>@polaris.alcf.anl.gov:/eagle/projects/PIGAN/<user>/ks_planet/code/planet/ks_dataset/
```

---

## Stage 3 — Data Conversion (cluster, ~30 min)

```bash
cd /eagle/projects/PIGAN/$USER/ks_planet/code/planet/argonne/
qsub 01_convert.pbs
qstat -u $USER   # watch status: Q (queued) → R (running) → F (finished)
```

Tail the log while it runs:
```bash
tail -f /eagle/projects/PIGAN/$USER/ks_planet/logs/convert.out
```

### Expected outputs

| Location | What | Size |
|---|---|---|
| `ks_dataset/train_episodes/` | 4096 KS train NPZ files | ~7 GB |
| `ks_dataset/test_episodes/` | 512 KS test NPZ files | ~4 GB |
| `ks_dataset/kolmogorov_train_episodes/` | 20000 Kolmogorov pairs | ~500 MB |
| `ks_dataset/kolmogorov_val_episodes/` | 5000 pairs | ~125 MB |
| `ks_dataset/kolmogorov_test_episodes/` | 5000 pairs | ~125 MB |

### W&B runs produced (2 of them)

1. **`ks_data_conversion`** (tag: `ks`) — should finish in ~10 min, show:
   - Scalars: `ks_train/episodes_done`, `ks_train/conversion_rate`
   - Tables: `ks_train_failures`, `ks_test_failures` (both should be empty)
   - Summary: `ks/conversion_status = PASSED`

2. **`kolmogorov_data_conversion`** (tag: `kolmogorov`) — should finish in ~20 min, show:
   - Scalars: `kolmogorov_train/episodes_done`, etc.
   - Tables: `kolmogorov_*_failures`
   - Config: `is_sequential = True/False`, `sequential_diff = <float>`
   - Summary: `kolmogorov/conversion_status = PASSED`

Filter the W&B project by tag `data_conversion` to see only these runs.

If `is_sequential = True`, edit `03_train_kolmogorov.pbs` and override:
```bash
qsub -v BATCH=50,LEN=40,CTX=20,MAXLEN=51 03_train_kolmogorov.pbs
```

---

## Stage 4 — Training (cluster, 12–48 hours per system)

Submit both:
```bash
qsub 02_train_ks.pbs
qsub 03_train_kolmogorov.pbs
```

Or chain submissions so a second job auto-starts when the first hits walltime:
```bash
KS_JOB=$(qsub 02_train_ks.pbs)
qsub -W depend=afterany:${KS_JOB} 02_train_ks.pbs   # picks up from checkpoint
```

### Scalars to monitor (W&B → `Physics_Sim/KS_PlaNet` → filter tag `training`)

| Scalar | Healthy trend | What it means |
|---|---|---|
| `train/objectives_image` | rises, plateaus near 0 | Image reconstruction log-likelihood |
| `train/objectives_divergence` | falls then stabilises around 2–10 | KL(posterior ‖ prior) |
| `train/closedloop_state_prior_entropy` | > posterior entropy | Sanity: prior is wider than posterior |
| `train/closedloop_state_posterior_entropy` | < prior entropy | Encoder is informative |
| `train/dist_summaries_image_abs_error` | falls below 0.05 | Per-pixel reconstruction error |
| `test/mean_score` | rises slowly, with noise | Open-loop prediction quality |
| `train/general_steps_per_second` | stable (~3–5 on A100) | Throughput |

### Image panels in W&B (PlaNet writes these every `log_every` steps)

Look for these media panels under each run:
- `closedloop/posterior/image` — reconstructions WITH access to observations
- `closedloop/prior/image` — reconstructions WITHOUT access (the world model)
- `openloop/image` — long rollout from latent state

Each panel is a strip: top row = ground truth, bottom row = model output.

### Expected training curves (KS, 1M steps)

```
objectives_image                   objectives_divergence
        ╱╲                                  ╲
       ╱  ╲   plateau                        ╲___
      ╱    ──────                                ─────  plateau
     ╱
    ╱
   0          1M steps             0           1M steps
```

If `objectives_divergence` stays at 0 or `posterior_entropy ≥ prior_entropy`,
the posterior is collapsing. Reduce `free_nats` to 1.0:
```bash
qsub -v MAX_STEPS=2000000 \
     -v EXTRA="free_nats: 1.0" 02_train_ks.pbs   # (would require minor PBS edit)
```

---

## Stage 5 — Comparing both runs

Once both training runs have ≥ 100k steps:

1. **Open both runs** in W&B's comparison view.
2. Group by `tag` (`ks` vs `kolmogorov`).
3. Compare:
   - `train/objectives_image` — does the 2D Kolmogorov data converge faster?
   - `test/mean_score` — does long-horizon KS beat one-step Kolmogorov on its own test set?
   - `train/general_steps_per_second` — what does the smaller chunk size buy you in throughput?
4. **Pull the open-loop image strips** from both runs into your paper figure.

### Plots to put in a report or paper

| Plot | Source | What it shows |
|---|---|---|
| Training loss curves (KS vs Kolmogorov) | W&B | Convergence comparison |
| Open-loop prediction strips at 5, 10, 20 steps | `openloop/image` panel | Rollout quality |
| Frequency-spectrum analysis of ground truth vs predicted | manual post-hoc (use `visualize_both.py` as starting point) | Diagnoses whether the model misses high-k modes (PDE-Refiner's finding) |
| Pearson correlation vs rollout time | manual post-hoc | The standard PDE-Refiner / LNS metric |

---

## Compute budget

You have 3,887 Polaris hours remaining (PIGAN). Rough estimates per
single-A100 job:

| Task | Steps | Walltime | Polaris hours used |
|---|---|---|---|
| Both conversions | n/a | 30 min | 0.5 h |
| KS training (1M steps) | 1,000,000 | ~10 h | 10 h |
| Kolmogorov training (1M steps) | 1,000,000 | ~8 h | 8 h |
| 5 hyperparameter variants × both datasets | — | — | ~100 h |

So you can afford 30+ training runs without burning through the allocation.

---

## Switching to Sophia (DGX-A100, 1,000 hr allocation)

If Polaris queue is long, Sophia is faster per node (8× A100). Changes:

1. `#PBS -l select=1:system=sophia` instead of `polaris`
2. Sophia uses `/grand` or `/eagle` similarly
3. Same NGC container works
4. Single-GPU usage wastes 7/8 of a Sophia node — only worth it for short
   debug jobs or if Polaris is congested.

---

## Cheat sheet

```bash
# Cluster
ssh <user>@polaris.alcf.anl.gov
cd /eagle/projects/PIGAN/$USER/ks_planet/code/planet
qstat -u $USER             # see your jobs
qdel <jobid>               # cancel
tail -f ../../logs/*.out   # watch logs

# Local dry-run
cd planet/
bash dry_run/local_pipeline.sh

# W&B
# https://wandb.ai/Physics_Sim/KS_PlaNet
# filter: tag in [ks, kolmogorov, data_conversion, training]
```
