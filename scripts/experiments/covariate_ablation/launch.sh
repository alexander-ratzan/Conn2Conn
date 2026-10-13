#!/bin/bash
# Spec v3 E1: one (direction, SC metric) x seed cell per array task (array index = seed); trains the listed conditions
# in one process (data loaded once). Outputs runs/<sc_metric>/<direction>/<condition>/seed<S>.json; done conditions
# are skipped, so a requeue resumes.
#   sbatch --job-name=e1_cov_<dir>_<metric> --array=0-4 --time=<t> launch.sh <direction> <sc_metric> "<conditions|all>"
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
DIRECTION="${1:?direction}"; SC_METRIC="${2:?sc_metric}"; CONDITIONS="${3:-all}"
SEED=${SLURM_ARRAY_TASK_ID}
module purge
REPO=/scratch/asr655/neuroinformatics/Conn2Conn
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg
echo "Starting ${SLURM_JOB_ID} (task ${SEED}) on $(hostname) at $(date); HEAD $(git -C ${REPO} rev-parse --short HEAD)"
echo "Campaign=e1_cov Direction=${DIRECTION} SC_metric=${SC_METRIC} Seed=${SEED} Conditions=${CONDITIONS}"
singularity exec --nv --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc "
    source /ext3/env.sh; export PYTHONUNBUFFERED=1; cd ${REPO}
    python scripts/experiments/covariate_ablation/run_cell.py --direction ${DIRECTION} --sc_metric ${SC_METRIC} \
      --seed ${SEED} --conditions ${CONDITIONS}"
echo "Job Over at $(date)"
