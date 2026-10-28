#!/bin/bash
# Spec v4 E4.0c: baselines re-selected on val average rank (D9) at k = 256, 512 and full rank. 15 tasks: index // 5 = k,
# index % 5 = seed (CPU). Then: python analyze_e40c.py (D10 code-size decision).
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=4:00:00
#SBATCH --array=0-14
#SBATCH --job-name=e4_e40c
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
KS=(256 512 full)
K=${KS[$((SLURM_ARRAY_TASK_ID / 5))]}; SEED=$((SLURM_ARRAY_TASK_ID % 5))
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
echo "Starting ${SLURM_JOB_ID} seed ${SEED} k ${K} on $(hostname) at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; cd ${CONN2CONN_DIR}/scripts/experiments/cross_condition_translation; python -u run_e40c.py --seed ${SEED} --k ${K}"
