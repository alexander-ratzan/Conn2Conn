#!/bin/bash
# Spec v4 E5: render the notebook (execute in place, outputs baked). It only loads E5's tables / results; no model fitting.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --job-name=e5_render
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.err
set -euo pipefail
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; python -I /scratch/asr655/tools/run_notebook.py ${CONN2CONN_DIR}/scripts/notebooks/model_overviews/cca_analysis_overview.ipynb"
