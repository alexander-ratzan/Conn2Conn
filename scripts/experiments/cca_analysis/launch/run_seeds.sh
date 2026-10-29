#!/bin/bash
# Spec v4 E5.1: representations, full-rank CCA, H1, H2 and G0, one seed per array task (CPU). Pilot: --array=0.
#   sbatch --array=0 scripts/experiments/cca_analysis/launch/run_seeds.sh ; then --array=1-4
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --array=0-4
#SBATCH --requeue
#SBATCH --job-name=e5_seed
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
echo "Starting ${SLURM_JOB_ID} seed ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8; cd ${CONN2CONN_DIR}/scripts/experiments/cca_analysis; python -u run_seed.py --seed ${SLURM_ARRAY_TASK_ID}"
