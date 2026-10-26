#!/bin/bash
# Spec v4 E2.1e: tables + figures from all seed records (CPU). Submitted with afterok on the seed arrays.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --job-name=e21_analyze
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.err
set -euo pipefail
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc \
  "source /ext3/env.sh; cd ${CONN2CONN_DIR}; python -u scripts/experiments/joint_flow_matching/e21_analyze.py"
