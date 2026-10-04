#!/bin/bash
# Validation for the fc-conditions branch (scripts/sbatch/checks/verify_fc_conditions.py), one task per parcellation.
# Compares against main's data/ + models/ at OLD_COMMIT. Submit from the branch checkout:
#   sbatch scripts/sbatch/checks/verify_fc_conditions.sh
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=verify_fc_conditions
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/verify_fc_conditions_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/verify_fc_conditions_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

module purge
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
OLD_COMMIT="${OLD_COMMIT:-cc78287}"
cd "${REPO_DIR}"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

PARCS=(Glasser 4S456Parcels)
PARC=${PARCS[$SLURM_ARRAY_TASK_ID]}
OLD_TREE="${SLURM_TMPDIR:-/tmp/${USER}_verify_fc_${SLURM_JOB_ID}}/old_tree"
mkdir -p "${OLD_TREE}"
git -C "${REPO_DIR}" archive "${OLD_COMMIT}" data models | tar -x -C "${OLD_TREE}"

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID} = ${PARC}) on $(hostname) at $(date)"
echo "Branch tree=${REPO_DIR} ($(git -C "${REPO_DIR}" rev-parse --short HEAD))  old=${OLD_COMMIT}"

singularity exec \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OLD_TREE=${OLD_TREE}
    cd ${REPO_DIR}
    python scripts/sbatch/checks/verify_fc_conditions.py ${PARC}
  "

echo "Job Over at $(date)"
