#!/bin/bash
# Submit the full KS-PlaNet Polaris pipeline from the Polaris login node.
#
# Run on Polaris after syncing:
#   cd /eagle/projects/PIGAN/$USER/ks_planet/code/planet
#   bash argonne/submit_pipeline.sh
#
# Optional overrides:
#   PROJECT=PIGAN SKIP_SETUP=1 MAX_STEPS=2000000 bash argonne/submit_pipeline.sh

set -euo pipefail

PROJECT="${PROJECT:-PIGAN}"
SKIP_SETUP="${SKIP_SETUP:-0}"
MAX_STEPS="${MAX_STEPS:-2000000}"
BATCH="${BATCH:-64}"
LEN="${LEN:-2}"
CTX="${CTX:-1}"
MAXLEN="${MAXLEN:-2}"

PROJ_DIR="/eagle/projects/${PROJECT}/${USER}/ks_planet"
REPO_DIR="${PROJ_DIR}/code"
CODE_DIR="${REPO_DIR}/planet"
SIF="${PROJ_DIR}/containers/ks-planet.sif"

cd "${CODE_DIR}"
mkdir -p "${PROJ_DIR}/logs" "${PROJ_DIR}/checkpoints" "${PROJ_DIR}/figures"

echo "Submitting KS-PlaNet pipeline"
echo "  Project   : ${PROJECT}"
echo "  Repo dir  : ${REPO_DIR}"
echo "  Code dir  : ${CODE_DIR}"
echo "  Container : ${SIF}"
echo "  MAX_STEPS : ${MAX_STEPS}"
echo

if [ "${SKIP_SETUP}" = "1" ] && [ -f "${SIF}" ]; then
    SETUP_JOB=""
    echo "Skipping setup because SKIP_SETUP=1 and container exists."
else
    SETUP_JOB="$(qsub argonne/00_setup_polaris.pbs)"
    echo "Setup job       : ${SETUP_JOB}"
fi

if [ -n "${SETUP_JOB:-}" ]; then
    CONVERT_JOB="$(qsub -W depend=afterok:${SETUP_JOB} argonne/01_convert.pbs)"
else
    CONVERT_JOB="$(qsub argonne/01_convert.pbs)"
fi
echo "Convert job     : ${CONVERT_JOB}"

KS_JOB="$(qsub -W depend=afterok:${CONVERT_JOB} -v MAX_STEPS=${MAX_STEPS} argonne/02_train_ks.pbs)"
KO_JOB="$(qsub -W depend=afterok:${CONVERT_JOB} -v MAX_STEPS=${MAX_STEPS},BATCH=${BATCH},LEN=${LEN},CTX=${CTX},MAXLEN=${MAXLEN} argonne/03_train_kolmogorov.pbs)"
echo "KS train job    : ${KS_JOB}"
echo "Kolmogorov job  : ${KO_JOB}"

BACKUP_JOB="$(qsub -W depend=afterany:${KS_JOB}:${KO_JOB} argonne/04_backup.pbs)"
echo "Backup job      : ${BACKUP_JOB}"
echo
echo "Monitor:"
echo "  qstat -u ${USER}"
echo "  tail -f ${PROJ_DIR}/logs/convert.out"
echo "  tail -f ${PROJ_DIR}/logs/train_ks.out"
echo "  tail -f ${PROJ_DIR}/logs/train_kolmogorov.out"
