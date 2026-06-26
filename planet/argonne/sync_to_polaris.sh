#!/bin/bash
# Sync this repo to Polaris Eagle storage.
#
# Run from your local WSL machine, from either the repo root or planet/:
#   POLARIS_USER=<alcf_username> PROJECT=PIGAN bash argonne/sync_to_polaris.sh
#   POLARIS_USER=<alcf_username> PROJECT=PIGAN SKIP_DATA=1 bash argonne/sync_to_polaris.sh
#
# Notes:
# - This uses rsync over SSH. For very large raw datasets, Globus is preferred,
#   but rsync is convenient for code plus moderate files.
# - Raw ks_dataset/*.npz and *.npy files are included unless SKIP_DATA=1.
# - Generated episodes, dry-run checkpoints, figures, and caches are excluded.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${CODE_ROOT}/../Dockerfile" ] && [ -f "${CODE_ROOT}/../setup.py" ]; then
    REPO_ROOT="$(cd "${CODE_ROOT}/.." && pwd)"
else
    REPO_ROOT="${CODE_ROOT}"
fi

POLARIS_USER="${POLARIS_USER:-${USER}}"
PROJECT="${PROJECT:-PIGAN}"
REMOTE_HOST="${POLARIS_USER}@polaris.alcf.anl.gov"
REMOTE_BASE="${REMOTE_BASE:-/eagle/projects/${PROJECT}/${POLARIS_USER}/ks_planet}"
REMOTE_CODE="${REMOTE_BASE}/code"
SKIP_DATA="${SKIP_DATA:-0}"

echo "Local repo : ${REPO_ROOT}"
echo "Remote     : ${REMOTE_HOST}:${REMOTE_CODE}/"
echo "Project    : ${PROJECT}"
echo "Skip data  : ${SKIP_DATA}"
echo

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_CODE}' '${REMOTE_BASE}/logs' '${REMOTE_BASE}/containers' '${REMOTE_BASE}/checkpoints' '${REMOTE_BASE}/figures'"

EXCLUDES=(
    --exclude '.git/'
    --exclude '.idea/'
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude 'planet/ks_dataset/train_episodes/'
    --exclude 'planet/ks_dataset/test_episodes/'
    --exclude 'planet/ks_dataset/kolmogorov_train_episodes/'
    --exclude 'planet/ks_dataset/kolmogorov_val_episodes/'
    --exclude 'planet/ks_dataset/kolmogorov_test_episodes/'
    --exclude 'planet/ks_dataset/dryrun_ks/'
    --exclude 'planet/ks_dataset/dryrun_kolmogorov/'
    --exclude 'planet/ks_dataset/figures/'
    --exclude 'planet/ks_dataset/logs/'
)

if [ "${SKIP_DATA}" = "1" ]; then
    EXCLUDES+=(
        --exclude 'planet/ks_dataset/*.npz'
        --exclude 'planet/ks_dataset/*.npy'
    )
fi

rsync -avzP --delete \
    "${EXCLUDES[@]}" \
    "${REPO_ROOT}/" \
    "${REMOTE_HOST}:${REMOTE_CODE}/"

echo
echo "Sync complete."
echo "On Polaris:"
echo "  cd ${REMOTE_CODE}/planet"
echo "  bash argonne/submit_pipeline.sh"
