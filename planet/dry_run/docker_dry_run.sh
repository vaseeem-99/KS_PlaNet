#!/bin/bash
# ============================================================================
# docker_dry_run.sh -- End-to-end dry-run on WSL2 with Docker
# ============================================================================

set -euo pipefail

# Resolve both common layouts:
#   repo root:  Dockerfile, setup.py, planet/
#   code root:  planet/ks_dataset, planet/ks_scripts, ...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -f "${CODE_ROOT}/Dockerfile" ] && [ -f "${CODE_ROOT}/setup.py" ]; then
    PROJECT_ROOT="${CODE_ROOT}"
    CODE_PREFIX=""
elif [ -f "${CODE_ROOT}/../Dockerfile" ] && [ -f "${CODE_ROOT}/../setup.py" ]; then
    PROJECT_ROOT="$(cd "${CODE_ROOT}/.." && pwd)"
    CODE_PREFIX="$(basename "${CODE_ROOT}")"
else
    echo "ERROR: Could not find Dockerfile/setup.py beside or above ${CODE_ROOT}"
    echo "Run from a complete repo checkout, for example: ~/planet/planet"
    exit 1
fi

CONTAINER_CODE_DIR="/workspace"
if [ -n "${CODE_PREFIX}" ]; then
    CONTAINER_CODE_DIR="/workspace/${CODE_PREFIX}"
fi

DATA_DIR="${PROJECT_ROOT}"
if [ -n "${CODE_PREFIX}" ]; then
    DATA_DIR="${PROJECT_ROOT}/${CODE_PREFIX}"
fi
DATA_DIR="${DATA_DIR}/ks_dataset"
LOG_DIR="${DATA_DIR}/logs"
IMAGE_NAME="ks-planet:v1" # FIXED: Matches your exact local tag name
DOCKER_BASE="nvcr.io/nvidia/tensorflow:20.10-tf1-py3"
HOST_UID="${SUDO_UID:-$(id -u)}"
HOST_GID="${SUDO_GID:-$(id -g)}"
DOCKER_USER_ARGS=(--user "${HOST_UID}:${HOST_GID}" -e HOME=/tmp)

restore_full_datasets() {
    if [ -d "${DATA_DIR}" ]; then
        cd "${DATA_DIR}"
        for f in *.full; do
            [ -e "$f" ] || continue
            mv "$f" "${f%.full}"
        done
        cd "${PROJECT_ROOT}"
    fi
}
trap restore_full_datasets EXIT

echo "============================================"
echo "KS-PlaNet local Docker dry-run"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Code dir     : ${CODE_ROOT}"
echo "  Image        : ${IMAGE_NAME}"
echo "============================================"

# ── 1. Pre-flight check: source files present ───────────────────────────────
REQUIRED=(
    "${DATA_DIR}/ks_train.npz" "${DATA_DIR}/ks_test.npz"
    "${DATA_DIR}/X_train.npy"  "${DATA_DIR}/Y_train.npy"
    "${DATA_DIR}/X_test.npy"   "${DATA_DIR}/Y_test.npy"
    "${DATA_DIR}/X_val.npy"    "${DATA_DIR}/Y_val.npy"
    "${PROJECT_ROOT}/Dockerfile" "${PROJECT_ROOT}/setup.py"
)
for f in "${REQUIRED[@]}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file missing: $f"
        exit 1
    fi
done
echo "[1/6] Required files present."
mkdir -p "${LOG_DIR}"

# ── 2. Precheck: Skip if already built ──────────────────────────────────────
cd "${PROJECT_ROOT}"
if docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "[2/6] Found existing image ${IMAGE_NAME}. Skipping build step entirely!"
else
    echo "[2/6] Image not found. Building Docker image (~1-2 minutes)..."
    docker build -t "${IMAGE_NAME}" .
fi

# ── 3. Build the 5-trajectory subset ────────────────────────────────────────
echo "[3/6] Creating 5-trajectory dry-run subset..."

docker run --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -v "${PROJECT_ROOT}":/workspace \
    -e KS_TRAIN_N=5 -e KS_TEST_N=5 \
    -e KO_TRAIN_N=5 -e KO_VAL_N=5 -e KO_TEST_N=5 \
    "${IMAGE_NAME}" \
    python -c "
import os, sys
sys.path.insert(0, '${CONTAINER_CODE_DIR}/dry_run')
import make_subset
make_subset.KS_TRAIN_N = int(os.environ.get('KS_TRAIN_N', 5))
make_subset.KS_TEST_N  = int(os.environ.get('KS_TEST_N',  5))
make_subset.KO_TRAIN_N = int(os.environ.get('KO_TRAIN_N', 5))
make_subset.KO_VAL_N   = int(os.environ.get('KO_VAL_N',   5))
make_subset.KO_TEST_N  = int(os.environ.get('KO_TEST_N',  5))
make_subset.main()
" 2>&1 | tee "${LOG_DIR}/dryrun_subset.log"

# Swap subsets into place
cd "${DATA_DIR}"
for orig in ks_train.npz ks_test.npz X_train.npy X_test.npy X_val.npy \
            Y_train.npy Y_test.npy Y_val.npy; do
    sub="${orig%.*}_subset.${orig##*.}"
    [ ! -f "${orig}.full" ] && mv "${orig}" "${orig}.full"
    cp "${sub}" "${orig}"
done
cd "${CODE_ROOT}"

# Clear converted episodes from any previous full or dry run. The converters
# skip existing files, so stale episodes would otherwise contaminate the subset.
rm -rf "${DATA_DIR}/train_episodes" "${DATA_DIR}/test_episodes" \
       "${DATA_DIR}/kolmogorov_train_episodes" \
       "${DATA_DIR}/kolmogorov_val_episodes" \
       "${DATA_DIR}/kolmogorov_test_episodes" \
       "${DATA_DIR}/figures"

# ── 4. Run KS conversion in container ───────────────────────────────────────
echo "[4/6] Running KS conversion on subset (in container)..."
docker run --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -v "${PROJECT_ROOT}":/workspace \
    -w "${CONTAINER_CODE_DIR}" \
    -e PYTHONPATH=/workspace \
    -e CONVERT_WORKERS=1 \
    "${IMAGE_NAME}" \
    python ks_scripts/convert_ks_to_planet.py \
    2>&1 | tee "${LOG_DIR}/dryrun_ks_convert.log"

# ── 5. Run Kolmogorov conversion in container ───────────────────────────────
echo "[5/6] Running Kolmogorov conversion on subset (in container)..."
docker run --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -v "${PROJECT_ROOT}":/workspace \
    -w "${CONTAINER_CODE_DIR}" \
    -e PYTHONPATH=/workspace \
    -e CONVERT_WORKERS=1 \
    "${IMAGE_NAME}" \
    python ks_scripts/convert_kolmogorov_to_planet.py \
    2>&1 | tee "${LOG_DIR}/dryrun_kolmogorov_convert.log"

echo "[5b/6] Creating converted-episode diagnostic plots..."
docker run --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -v "${PROJECT_ROOT}":/workspace \
    -w "${CONTAINER_CODE_DIR}" \
    -e PYTHONPATH=/workspace \
    "${IMAGE_NAME}" \
    python scripts/plot_episode_diagnostics.py \
        --dataset ks \
        --data-dir ks_dataset \
        --outdir ks_dataset/figures/ks_diagnostics \
    2>&1 | tee "${LOG_DIR}/dryrun_ks_diagnostics.log"

docker run --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -v "${PROJECT_ROOT}":/workspace \
    -w "${CONTAINER_CODE_DIR}" \
    -e PYTHONPATH=/workspace \
    "${IMAGE_NAME}" \
    python scripts/plot_episode_diagnostics.py \
        --dataset kolmogorov \
        --data-dir ks_dataset \
        --outdir ks_dataset/figures/kolmogorov_diagnostics \
    2>&1 | tee "${LOG_DIR}/dryrun_kolmogorov_diagnostics.log"

# ── 6. Run 100 training steps for each dataset ──────────────────────────────
echo "[6/6] Running 100 training steps for each dataset (in container)..."

# KS dry training
docker run --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -v "${PROJECT_ROOT}":/workspace \
    -w "${CONTAINER_CODE_DIR}" \
    -e PYTHONPATH=/workspace \
    -e CUDA_VISIBLE_DEVICES="" \
    "${IMAGE_NAME}" \
    python -m scripts.train \
        --logdir ks_dataset/dryrun_ks \
        --config ks_config \
        --params "{tasks: [ks_chaos], max_steps: 100, train_steps: 50, test_steps: 10, train_dir: ${CONTAINER_CODE_DIR}/ks_dataset/train_episodes, test_dir: ${CONTAINER_CODE_DIR}/ks_dataset/test_episodes}" \
    2>&1 | tee "${LOG_DIR}/dryrun_ks_train.log"

# Kolmogorov dry training
docker run --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -v "${PROJECT_ROOT}":/workspace \
    -w "${CONTAINER_CODE_DIR}" \
    -e PYTHONPATH=/workspace \
    -e CUDA_VISIBLE_DEVICES="" \
    "${IMAGE_NAME}" \
    python -m scripts.train \
        --logdir ks_dataset/dryrun_kolmogorov \
        --config kolmogorov_config \
        --params "{tasks: [kolmogorov_flow], max_steps: 100, train_steps: 50, test_steps: 10, train_dir: ${CONTAINER_CODE_DIR}/ks_dataset/kolmogorov_train_episodes, test_dir: ${CONTAINER_CODE_DIR}/ks_dataset/kolmogorov_test_episodes}" \
    2>&1 | tee "${LOG_DIR}/dryrun_kolmogorov_train.log"

# Restore original data files
echo ""
echo "Converted image previews:"
find "${DATA_DIR}/figures" -maxdepth 2 -type f -name '*.png' | sort || true
echo ""
echo "Console logs:"
find "${LOG_DIR}" -maxdepth 1 -type f -name '*.log' | sort || true
echo ""
echo "Restoring full datasets..."
restore_full_datasets
trap - EXIT

echo ""
echo "============================================"
echo "Dry-run PASSED. Pipeline works end-to-end."
echo "============================================"
