#!/bin/bash
# Composite-loss protocol (spec v2 E1). E1.5 report: tables + figures from the instance's runs (CPU only).
#   sbatch scripts/experiments/composite_loss/launch_report.sh <instance>      (instance = folder under scripts/experiments/composite_loss/)
# Exit code 2 from protocol.py = a D3 stop condition tripped (see the job log and the instance state.yml).
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --mem=32GB
#SBATCH --requeue
#SBATCH --job-name=e1_report
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/e1_report_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/e1_report_%A_%a.err
#SBATCH --array=0

set -euo pipefail

INSTANCE="${1:?usage: sbatch scripts/experiments/composite_loss/launch_report.sh <instance>}"
module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SLURM_ARRAY_TASK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT}) step=report instance=${INSTANCE} on $(hostname) at $(date)"

singularity exec \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
    cd ${CONN2CONN_DIR}
    INSTANCE=${INSTANCE} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0} SLURM_ARRAY_TASK_COUNT=${SLURM_ARRAY_TASK_COUNT}
    python scripts/experiments/composite_loss/protocol.py report --instance ${INSTANCE}
  "

echo "Job Over at $(date)"
