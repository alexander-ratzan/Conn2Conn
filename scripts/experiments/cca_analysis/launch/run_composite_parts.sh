#!/bin/bash
# Spec v4 E5.2 (H3) split into parts so each task stays under 2 h on any GPU. Array index = seed_index * NPARTS + part.
#   sbatch --array=0-3 scripts/experiments/cca_analysis/launch/run_composite_parts.sh "1 4" 2   # seeds 1, 4; 2 parts each
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --job-name=e5_composite
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
SEEDS=(${1:?seeds, e.g. "1 4"}); NPARTS="${2:-2}"
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / NPARTS))]}; PART=$((SLURM_ARRAY_TASK_ID % NPARTS))
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
echo "Starting ${SLURM_JOB_ID} seed ${SEED} part ${PART}/${NPARTS} on $(hostname) at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
singularity exec --nv --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; cd ${CONN2CONN_DIR}/scripts/experiments/cca_analysis; python -u run_composite.py --seed ${SEED} --part ${PART} --nparts ${NPARTS}"
