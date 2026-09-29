#!/bin/bash
# Array: real-data verification for the modeling track (spec §8), one task per index:
#   0 = dev_runs (M8)   1 = cross_tree (M7a + M5)   2 = tune (M6)
# Reports: results/logs/verify_modeling_track_<task>.json
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=verify_modeling_track
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/verify_modeling_track_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/verify_modeling_track_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TUNE_CPUS_PER_TRIAL=4
export TUNE_GPUS_PER_TRIAL=1
export MPLBACKEND=Agg

TASKS=(dev_runs cross_tree tune)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}

# Old code for cross-tree comparisons: pre-M7 (still has residual_mode none) and pre-M5 (scalar reg).
OLD_TREES="${SLURM_TMPDIR:-/tmp/${USER}_verify_$SLURM_JOB_ID}/old_trees"
mkdir -p "${OLD_TREES}/pre_m7" "${OLD_TREES}/pre_m5"
git -C "${CONN2CONN_DIR}" archive f74cc0e models | tar -x -C "${OLD_TREES}/pre_m7"
git -C "${CONN2CONN_DIR}" archive 5286526 models | tar -x -C "${OLD_TREES}/pre_m5"

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID} = ${TASK}) on $(hostname) at $(date)"

singularity exec --nv \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OLD_TREES=${OLD_TREES}
    export TUNE_CPUS_PER_TRIAL=${TUNE_CPUS_PER_TRIAL} TUNE_GPUS_PER_TRIAL=${TUNE_GPUS_PER_TRIAL}
    unset RAY_TMPDIR
    cd ${CONN2CONN_DIR}
    python scripts/sbatch/checks/verify_modeling_track.py ${TASK}
  "

echo "Job Over at $(date)"
