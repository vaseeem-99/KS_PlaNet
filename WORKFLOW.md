# WORKFLOW — KS-PlaNet on WSL2 (dry-run) → Polaris (production)

Single document that walks the entire pipeline end-to-end. Read top to bottom
before running anything.

---

## The big picture

```
   ┌──────────────────────────────────────────────────────────────┐
   │  PHASE 1 — WSL2 dry-run (your laptop, ~20 min)               │
   │                                                              │
   │   docker build  →  docker run with subset (5 traj each)      │
   │   verify pipeline end-to-end                                 │
   └──────────────────────┬───────────────────────────────────────┘
                          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  PHASE 2 — Transfer to Polaris                               │
   │                                                              │
   │   rsync code  →  /eagle/projects/PIGAN/$USER/ks_planet/code/ │
   │   rsync data  →  /eagle/projects/PIGAN/$USER/ks_planet/code/ │
   │                  planet/ks_dataset/                          │
   └──────────────────────┬───────────────────────────────────────┘
                          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  PHASE 3 — Polaris setup (PBS job, ~30 min, one-time)        │
   │                                                              │
   │   qsub 00_setup_polaris.pbs                                  │
   │   → builds Apptainer .sif from Dockerfile on a compute node  │
   └──────────────────────┬───────────────────────────────────────┘
                          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  PHASE 4 — Run the experiments                               │
   │                                                              │
   │   qsub 01_convert.pbs                  (debug queue, 30 min) │
   │   qsub 02_train_ks.pbs                 (preemptable, 24h)    │
   │   qsub 03_train_kolmogorov.pbs         (preemptable, 24h)    │
   └──────────────────────┬───────────────────────────────────────┘
                          ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  PHASE 5 — Pull results back to laptop, generate plots       │
   │                                                              │
   │   rsync checkpoints + logs → laptop                          │
   │   python analyze_results.py → generates all research plots   │
   └──────────────────────────────────────────────────────────────┘
```

---

## Files in this repo

```
planet/
├── setup.py                        ← install with `pip install -e ".[container]"`
├── Dockerfile                      ← adds PlaNet deps to NGC TF1 image
├── WORKFLOW.md                     ← this file
├── PLOTS_GUIDE.md                  ← catalog of every plot we generate
├── analyze_results.py              ← post-training research plots
│
├── ks_dataset/                     ← raw + converted data (STAYS OUTSIDE CONTAINER)
│   ├── ks_train.npz, ks_test.npz
│   ├── X_*.npy, Y_*.npy
│   └── (*_episodes/ created by converters)
│
├── ks_scripts/
│   ├── convert_ks_to_planet.py
│   ├── convert_kolmogorov_to_planet.py
│   ├── visualize_ks.py
│   ├── visualize_kolmogorov.py
│   └── visualize_both.py
│
├── argonne/
│   ├── 00_setup_polaris.pbs        ← build Apptainer SIF (compute node)
│   ├── 01_convert.pbs              ← both converters, debug queue
│   ├── 02_train_ks.pbs             ← KS training, preemptable queue
│   └── 03_train_kolmogorov.pbs     ← Kolmogorov training, preemptable queue
│
├── dry_run/
│   ├── make_subset.py              ← 5-trajectory subset creator
│   └── docker_dry_run.sh           ← WSL2 Docker end-to-end test
│
├── scripts/  tools/  training/  models/  networks/  control/
│   (unchanged PlaNet code + small appends documented elsewhere)
```

---

## Phase 1 — WSL2 dry-run

**What it proves:** every converter, every reader, every PlaNet config works
end-to-end on tiny data, inside the same container you'll run on Polaris.

### Prerequisites

- WSL2 Ubuntu with Docker installed
- (Optional) NVIDIA Container Toolkit for GPU pass-through. Without it
  the dry-run will use CPU mode — fine for 100 steps.

### Run

```bash
cd planet/
bash dry_run/docker_dry_run.sh
```

The script:
1. Pulls `nvcr.io/nvidia/tensorflow:20.10-tf1-py3` (one-time, ~5 GB)
2. Builds `ks-planet:1.0` from the Dockerfile (adds PlaNet pip deps)
3. Backs up `ks_train.npz` etc. and swaps in 5-trajectory subsets
4. Runs KS conversion inside the container
5. Runs Kolmogorov conversion inside the container
6. Runs 100 PlaNet training steps for each dataset
7. Restores original data files

**Total time: ~20 min first run, ~3 min thereafter** (cached layers).

### Healthy output

```
[1/6] Required files present.
[2/6] Image ks-planet:1.0 already built. Skipping.
[3/6] Creating 5-trajectory dry-run subset...
[4/6] Running KS conversion on subset (in container)...
  [train] 5/5 (100%)  5 ok  0 failed
[5/6] Running Kolmogorov conversion on subset (in container)...
  [train] 5/5 (100%)  5 ok  0 failed
[6/6] Running 100 training steps for each dataset (in container)...
  ...
Dry-run PASSED. Pipeline works end-to-end.
```

### What "data stays outside container" looks like

The Docker run uses `-v $(pwd):/workspace` to bind-mount your project
directory. Anything written under `ks_dataset/` lands on your WSL2 disk,
not inside the container layer. The container is stateless — destroy it,
your data is fine.

---

## Phase 2 — Transfer to Polaris

```bash
# From your laptop
rsync -avzP \
    --exclude='ks_dataset/*_episodes' \
    --exclude='ks_dataset/*.full' \
    --exclude='ks_dataset/*_subset.*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    planet/ \
    <user>@polaris.alcf.anl.gov:/eagle/projects/PIGAN/<user>/ks_planet/code/planet/
```

This copies code + raw data (`.npz` + `.npy`) but skips the converted
episode directories — those are regenerated on Polaris in Phase 4.

> The raw data is ~10 GB (KS ~2 GB + Kolmogorov ~7 GB). At Polaris's
> network speeds, allow 10–30 min for the first transfer. Subsequent
> rsyncs only ship deltas.

---

## Phase 3 — Polaris setup (one-time)

Login, submit the setup job, wait for it to finish.

```bash
ssh <user>@polaris.alcf.anl.gov
cd /eagle/projects/PIGAN/$USER/ks_planet/code/planet/argonne
qsub 00_setup_polaris.pbs
qstat -u $USER   # status: Q queued, R running, F finished
```

The job:
1. Requests a debug-queue compute node (free, 1 hour max)
2. Loads `apptainer` via `ml use /soft/modulefiles; ml spack-pe-base; ml apptainer`
3. Pulls the NGC base image
4. Builds a custom SIF from the `Dockerfile` (adds PlaNet pip deps)
5. Verifies imports inside the container
6. `pip install -e .` the planet package (into your home so it persists)
7. Logs in to W&B (set `WANDB_API_KEY` env var before submit to skip interactive)

### Polaris-specific reasons this is a PBS job, not a shell script

> Per ALCF docs: *"Container build and execution are only supported on
> Polaris compute nodes."* You cannot run `apptainer build` on a login node.

### W&B login

Easiest: set the API key once, then re-submit setup:
```bash
export WANDB_API_KEY="your_key_from_wandb.ai/authorize"
qsub -v WANDB_API_KEY 00_setup_polaris.pbs
```

Or after setup finishes, login manually inside the SIF (also on a compute node):
```bash
qsub -I -A PIGAN -q debug -l select=1 -l walltime=00:30:00 -l filesystems=home:eagle
# once on the compute node:
ml use /soft/modulefiles && ml spack-pe-base && ml apptainer
apptainer exec /eagle/projects/PIGAN/$USER/ks_planet/containers/ks-planet.sif wandb login
```

---

## Phase 4 — Run the experiments

### Step 4a: data conversion

```bash
qsub 01_convert.pbs
```

- Queue: `debug` (free, 1 hr max)
- Runs both converters back-to-back (~30 min)
- Outputs land in `code/planet/ks_dataset/*_episodes/` (outside container, on /eagle)
- Logs go to `logs/convert.out`
- W&B: two runs tagged `data_conversion`

Watch progress:
```bash
qstat -u $USER
tail -f /eagle/projects/PIGAN/$USER/ks_planet/logs/convert.out
```

### Step 4b: training (one or both, in parallel)

```bash
qsub 02_train_ks.pbs                          # 24 h walltime on preemptable
qsub 03_train_kolmogorov.pbs                  # 24 h walltime on preemptable
```

- Queue: `preemptable` (1-10 nodes, 72h walltime, can be preempted)
- `#PBS -r y` makes jobs auto-restart if preempted by a `demand` job
- Backups every 30 min into `checkpoints/*_run_backup/snapshot_<ts>/`
- W&B: runs tagged `training` + (`ks` or `kolmogorov`)

To do shorter test runs (e.g., 100k steps):
```bash
qsub -v MAX_STEPS=100000 02_train_ks.pbs
```

To chain a follow-up job that picks up where the previous left off:
```bash
JOB_ID=$(qsub 02_train_ks.pbs)
qsub -W depend=afterany:${JOB_ID} 02_train_ks.pbs
```

### Why not `prod` queue

Per Polaris docs, `prod` requires **minimum 10 nodes**. Your single-GPU
training would never start there. `preemptable` is the right queue for
1-node jobs that need long walltime. `capacity` would also work (168h
walltime) but allows only 1 running job per user, so you couldn't run KS
and Kolmogorov in parallel.

---

## Phase 5 — Pull results back, generate plots

### Download checkpoints + W&B logs

```bash
# From laptop
mkdir -p ~/ks_planet_results
rsync -avzP \
    <user>@polaris.alcf.anl.gov:/eagle/projects/PIGAN/<user>/ks_planet/checkpoints/ \
    ~/ks_planet_results/checkpoints/

rsync -avzP \
    <user>@polaris.alcf.anl.gov:/eagle/projects/PIGAN/<user>/ks_planet/logs/ \
    ~/ks_planet_results/logs/
```

Checkpoint sizes: ~500 MB per run including event files.

### Generate publication-ready plots

Inside the Docker container locally (for matplotlib + plumbing):

```bash
docker run --rm \
    -v ~/ks_planet_results:/results \
    -v $(pwd):/workspace \
    ks-planet:1.0 \
    python /workspace/analyze_results.py \
        --logdir /results/checkpoints/ks_run \
        --dataset ks \
        --output /results/figures_ks

docker run --rm \
    -v ~/ks_planet_results:/results \
    -v $(pwd):/workspace \
    ks-planet:1.0 \
    python /workspace/analyze_results.py \
        --logdir /results/checkpoints/kolmogorov_run \
        --dataset kolmogorov \
        --output /results/figures_kolmogorov
```

See `PLOTS_GUIDE.md` for what each plot tells you.

---

## Recovering from common failures

| Symptom | Fix |
|---|---|
| `qstat: Unknown queue: prod` | You're not on Polaris. Try `polaris.alcf.anl.gov` |
| Job stuck in `Q` for hours | `qstat -f <jobid>` to see comments. preemptable queue can be slow |
| Job killed early ("preempted") | Has `#PBS -r y` → auto-restarts. Otherwise resubmit |
| `Apptainer: command not found` | Forgot `ml use /soft/modulefiles && ml apptainer` |
| `Permission denied` on container | Add `--fakeroot` to build, check `/home` quotas |
| W&B silently not logging | Verify `wandb login` ran inside the container, check proxy env vars |
| `OOM` on A100 | Reduce `batch_shape[0]` from 50 to 32 |

---

## Cheat sheet

```bash
# WSL2
cd planet/
bash dry_run/docker_dry_run.sh

# Transfer
rsync -avzP --exclude='*_episodes' planet/ <user>@polaris.alcf.anl.gov:/eagle/projects/PIGAN/<user>/ks_planet/code/planet/

# Polaris
ssh <user>@polaris.alcf.anl.gov
cd /eagle/projects/PIGAN/$USER/ks_planet/code/planet/argonne
qsub 00_setup_polaris.pbs     # one-time
qsub 01_convert.pbs            # ~30 min
qsub 02_train_ks.pbs           # ~24 h
qsub 03_train_kolmogorov.pbs   # ~24 h

# Monitor
qstat -u $USER
tail -f /eagle/projects/PIGAN/$USER/ks_planet/logs/*.out
# W&B: https://wandb.ai/Physics_Sim/KS_PlaNet

# Download
rsync -avzP <user>@polaris:/eagle/projects/PIGAN/<user>/ks_planet/checkpoints/ ./results/

# Analyse
python analyze_results.py --logdir ./results/checkpoints/ks_run --dataset ks --output ./figures_ks
```
