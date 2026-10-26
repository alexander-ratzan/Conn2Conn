#!/bin/bash
# Spec v4 E2.1: one seed x model per array task. Array index = model index; seeds passed as an argument.
#   sbatch --array=0-5 launch/e21_run.sh <seed>          models: refs corrector corrector_id bridge flow latent2sc (6 flow_free, 7 flow_linear: E2.1 addendum)
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --job-name=e21_run
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
SEED="${1:?seed}"
TARGET="${2:-cca}"                 # cca (E2.1) or pca (second addendum: block FC PCA targets)
MODELS=(refs corrector corrector_id bridge flow latent2sc flow_free flow_linear)
MODEL=${MODELS[${SLURM_ARRAY_TASK_ID}]}
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "Starting ${SLURM_JOB_ID} seed ${SEED} model ${MODEL} on $(hostname) at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
singularity exec --nv --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc \
  "source /ext3/env.sh; cd ${CONN2CONN_DIR}; python -u scripts/experiments/joint_flow_matching/e21_run.py --model ${MODEL} --seed ${SEED} --target ${TARGET}"
echo "Job over at $(date)"
