#!/bin/bash
# Spec v4 E2 GPU checks: G1 synthetic gate, then 2-trial smoke runs of A3 and A4 on seed 0 (G0 end-to-end item and
# the time per trial that sets the pilot / main job limits).
#   sbatch scripts/experiments/joint_flow_matching/launch/gpu_checks.sh
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --job-name=e1_gpu_checks
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%j.err
set -euo pipefail
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "Starting ${SLURM_JOB_ID} on $(hostname) at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
run() { singularity exec --nv --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc "source /ext3/env.sh; cd ${CONN2CONN_DIR}; $*"; }
E=scripts/experiments/joint_flow_matching
run "nvidia-smi --query-gpu=name,memory.total --format=csv"
run "python -u ${E}/checks/synthetic.py"
run "python -u ${E}/tune.py --arm A3 --seed 0 --n-trials 2 --tag smoke"
run "python -u ${E}/tune.py --arm A4 --seed 0 --n-trials 2 --tag smoke"
echo "Job over at $(date)"
