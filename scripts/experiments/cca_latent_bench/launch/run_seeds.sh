#!/bin/bash
# Spec v4 E1: one seed per array task (closed form, CPU). Pilot: sbatch --array=0 ... --check
#   sbatch --array=0 scripts/experiments/cca_latent_bench/launch/run_seeds.sh --check     # E1.1 pilot (gate G1)
#   sbatch --array=1-4 scripts/experiments/cca_latent_bench/launch/run_seeds.sh           # E1.2
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --job-name=e2_cca_bench
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "Starting ${SLURM_JOB_ID} seed ${SLURM_ARRAY_TASK_ID} $* at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; cd ${CONN2CONN_DIR}; python -u scripts/experiments/cca_latent_bench/run_seed.py --seed ${SLURM_ARRAY_TASK_ID} $*"
echo "Job over at $(date)"
