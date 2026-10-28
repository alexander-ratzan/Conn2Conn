#!/bin/bash
# windowed_fc_gain G7b: E2.1 flows on part-sample arms (GPU). Needs run_g7a.sh (arm caches).
# 50 tasks: index // 25 = arm group (0: none + halves arms, 1: quarters arms), (index % 25) // 5 = model x route,
# index % 5 = seed. A requeued task resumes after its last finished arm.
#   sbatch --array=0-49%10 scripts/experiments/windowed_fc_gain/launch/run_g7b.sh
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --job-name=wfg_g7b
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err
set -euo pipefail
MR=(flow:cca flow_linear:cca flow_linear:pca flow_free:cca flow_free:pca)
ARM_GROUPS=("none aug_halves token_halves parts_halves" "aug_quarters token_quarters parts_quarters"); TAGS=(halves quarters)
i=${SLURM_ARRAY_TASK_ID}
GROUP="${ARM_GROUPS[$((i / 25))]}"; TAG="${TAGS[$((i / 25))]}"; r=$((i % 25)); MODEL="${MR[$((r / 5))]%%:*}"; ROUTE="${MR[$((r / 5))]##*:}"; SEED=$((r % 5))
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
module purge
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
echo "Starting ${SLURM_JOB_ID} ${MODEL} ${ROUTE} seed ${SEED} arms ${GROUP} on $(hostname) at $(date) commit $(git -C ${CONN2CONN_DIR} rev-parse --short HEAD)"
singularity exec --nv --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash -lc \
  "source /ext3/env.sh; cd ${CONN2CONN_DIR}; python -u scripts/experiments/windowed_fc_gain/run_g7b.py --model ${MODEL} --route ${ROUTE} --seed ${SEED} --arms ${GROUP} --out-tag ${TAG}"
echo "Job over at $(date)"
