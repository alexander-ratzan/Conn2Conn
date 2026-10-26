#!/bin/bash
# Spec v4 E2 arm A1 (L1-L4 + references), CPU, one seed per array task, every k in config k_grid.
#   sbatch --array=0-4 scripts/experiments/joint_flow_matching/launch/linear_arms.sh [variant]
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --job-name=e1_linear
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
VARIANT="${1:-main}"
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "Starting ${SLURM_JOB_ID} seed ${SLURM_ARRAY_TASK_ID} variant ${VARIANT} at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; cd ${CONN2CONN_DIR}; python -u scripts/experiments/joint_flow_matching/linear_arms.py --seed ${SLURM_ARRAY_TASK_ID} --variant ${VARIANT}"
echo "Job over at $(date)"
