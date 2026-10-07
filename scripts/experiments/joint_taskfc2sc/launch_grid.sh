#!/bin/bash
# Spec v3 E2 grid (CPU): array task i -> version = [raw, predictive_resid][i / 5], seed = i % 5 (grid.py).
#   sbatch scripts/experiments/joint_taskfc2sc/launch_grid.sh            # all 10 tasks
#   sbatch --array=0-4 scripts/experiments/joint_taskfc2sc/launch_grid.sh  # raw only
# Reruns skip cells already in runs/<version>/seed<S>.csv.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --array=0-9
#SBATCH --job-name=e2_joint_grid
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
module purge
VERSIONS=(raw predictive_resid)
VERSION=${VERSIONS[$(( SLURM_ARRAY_TASK_ID / 5 ))]}
SEED=$(( SLURM_ARRAY_TASK_ID % 5 ))
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "Campaign=e2_joint Version=${VERSION} Seed=${SEED} Commit=$(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; cd ${CONN2CONN_DIR}; python scripts/experiments/joint_taskfc2sc/grid.py --seed ${SEED} --version ${VERSION}"
echo "Job Over at $(date)"
