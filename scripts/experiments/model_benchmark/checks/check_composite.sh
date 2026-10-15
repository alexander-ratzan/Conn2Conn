#!/bin/bash
# Spec v3 I1 acceptance (GPU, ~15 min): sbatch scripts/experiments/model_benchmark/checks/check_composite.sh
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=e3_check_composite
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.err
set -euo pipefail
module purge
echo "Starting ${SLURM_JOB_ID} on $(hostname) at $(date)"
singularity exec --nv --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; export PYTHONUNBUFFERED=1 MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8; cd /scratch/asr655/neuroinformatics/Conn2Conn; python scripts/experiments/model_benchmark/checks/check_composite.py"
echo "Job Over at $(date)"
