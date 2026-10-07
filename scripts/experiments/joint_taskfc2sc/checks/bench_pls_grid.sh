#!/bin/bash
# Spec v3 E2 feasibility benchmark (CPU): sbatch scripts/experiments/joint_taskfc2sc/checks/bench_pls_grid.sh
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --mem=48GB
#SBATCH --job-name=e2_bench_pls
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.err
set -euo pipefail
module purge
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
echo "Starting ${SLURM_JOB_ID} on $(hostname) at $(date)"
singularity exec --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "source /ext3/env.sh; cd /scratch/asr655/neuroinformatics/Conn2Conn; python scripts/experiments/joint_taskfc2sc/checks/bench_pls_grid.py"
echo "Job Over at $(date)"
