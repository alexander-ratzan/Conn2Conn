#!/bin/bash
# Composite-loss protocol (spec v2 E1). E1.3 re-check after a failed consensus check: each seed's own best Stage 1
# config retrained on its seed (parallel on one GPU); the consensus is re-checked against those retrained values
# (if it still misses: accepted as a fallback, recorded in state.yml consensus_check). Chain grid + report afterok.
#   sbatch scripts/experiments/composite_loss/launch_rebaseline.sh <instance>      (instance = folder under scripts/experiments/composite_loss/)
# Exit code 2 from protocol.py = a D3 stop condition tripped (see the job log and the instance state.yml).
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --job-name=e1_rebaseline
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/e1_rebaseline_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/e1_rebaseline_%A_%a.err
#SBATCH --array=0

set -euo pipefail

INSTANCE="${1:?usage: sbatch scripts/experiments/composite_loss/launch_rebaseline.sh <instance>}"
module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SLURM_ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT}) step=rebaseline instance=${INSTANCE} on $(hostname) at $(date)"

singularity exec --nv \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
    cd ${CONN2CONN_DIR}
    INSTANCE=${INSTANCE} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0} SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}
    set -e
    python scripts/experiments/composite_loss/protocol.py rebaseline --instance ${INSTANCE}
  "

echo "Job Over at $(date)"
