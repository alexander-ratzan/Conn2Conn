#!/bin/bash
# Spec v4 E5.3 compute: notebook section-1 inputs (run_descriptives.py) and the H1-H3 tables / figures (analyze.py), CPU.
#   sbatch [--dependency=afterok:<h3 job>] scripts/experiments/cca_analysis/launch/run_analysis.sh [--skip-descriptives]
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --job-name=e5_analysis
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.err
set -euo pipefail
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
DESC=1; [ "${1:-}" = "--skip-descriptives" ] && DESC=0
module purge
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8; cd ${CONN2CONN_DIR}/scripts/experiments/cca_analysis; if [ ${DESC} = 1 ]; then python -u run_descriptives.py; fi; python -u analyze.py"
