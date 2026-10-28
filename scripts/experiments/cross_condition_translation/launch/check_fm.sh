#!/bin/bash
# Spec v4 E4.2: flow optimisability checks + G1 (GPU). Args: k [--quick]
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name=e4_check_fm
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.err
set -euo pipefail
K="${1:-256}"; EXTRA="${2:-}"
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
echo "Starting ${SLURM_JOB_ID} k ${K} ${EXTRA} on $(hostname) at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
singularity exec --nv --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc \
  "source /ext3/env.sh; cd ${CONN2CONN_DIR}/scripts/experiments/cross_condition_translation; python -u checks/check_fm.py --k ${K} ${EXTRA}"
