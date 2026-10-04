#!/bin/bash
# Headless run of a notebook's code cells (scripts/sbatch/checks/run_notebook_cells.py). Submit from the repo root:
#   sbatch scripts/sbatch/checks/run_notebook_cells.sh scripts/notebooks/EDA/FC_matrix_analysis.ipynb SAVE_FIGURES=True
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=run_notebook_cells
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/run_notebook_cells_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/run_notebook_cells_%j.err

set -euo pipefail

module purge
REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${REPO_DIR}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

echo "Starting job ${SLURM_JOB_ID} on $(hostname) at $(date): $* ($(git rev-parse --short HEAD))"

singularity exec \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
    cd ${REPO_DIR}
    python scripts/sbatch/checks/run_notebook_cells.py $*
  "

echo "Job Over at $(date)"
