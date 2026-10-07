#!/bin/bash
# Spec v3 E0 pre-flight (CPU): check_fc_source_condition.py for each new step's parcellation x direction.
#   task 0: Glasser sc2fc seed 0 · task 1: 4S456Parcels fc2sc seeds 0 1 · task 2: 4S456Parcels sc2fc seed 0
#   sbatch scripts/experiments/task_fc2sc/checks/check_fc_source_condition.sh
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=96GB
#SBATCH --job-name=e0_check_fc_source
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail
module purge
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
case "${SLURM_ARRAY_TASK_ID}" in
  0) ARGS="--parcellation Glasser --direction sc2fc --seeds 0" ;;
  1) ARGS="--parcellation 4S456Parcels --direction fc2sc --seeds 0 1" ;;
  2) ARGS="--parcellation 4S456Parcels --direction sc2fc --seeds 0" ;;
esac
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "Starting ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: ${ARGS} at $(date)"
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; cd ${CONN2CONN_DIR}; python scripts/experiments/task_fc2sc/checks/check_fc_source_condition.py ${ARGS}"
echo "Job Over at $(date)"
