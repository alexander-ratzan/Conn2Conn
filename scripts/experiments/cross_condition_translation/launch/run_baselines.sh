#!/bin/bash
# Spec v4 E4.0: closed-form baselines, one seed per array task (CPU). Then:
#   sbatch --dependency=afterok:<jobid> scripts/experiments/cross_condition_translation/launch/analyze_baselines.sh
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=2:00:00
#SBATCH --array=0-4
#SBATCH --job-name=e4_baselines
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
K="${1:-}"                          # optional: block FC codes per condition (default config k), e.g. 512
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
echo "Starting ${SLURM_JOB_ID} seed ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; cd ${CONN2CONN_DIR}/scripts/experiments/cross_condition_translation; python -u run_baselines.py --seed ${SLURM_ARRAY_TASK_ID} ${K:+--k $K}"
