#!/bin/bash
# Spec v3 E0 task: one FC condition x one seed (array index = shuffle seed): Ray Tune over configs/<condition>.yml,
# then the best-trial report (its JSON summary in this task's log is what run.py reads). Submit through submit.py.
# Positional: <condition> <num_samples> <search_alg> <gpus_per_trial> <max_concurrent>
# CONN2CONN_DIR (env) = checkout to run from (default: main).
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err

set -euo pipefail
COND="${1:?condition}"; NUM_SAMPLES="${2:?num_samples}"; SEARCH_ALG="${3:?search_alg}"
GPUS_PER_TRIAL="${4:-0.25}"; MAX_CONCURRENT="${5:-4}"
SEED=${SLURM_ARRAY_TASK_ID}
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
CONFIG="scripts/experiments/task_fc_to_sc/configs/${COND}.yml"
CPUS_PER_TRIAL=$(( ${SLURM_CPUS_PER_TASK:-10} / ${MAX_CONCURRENT} )); [ "${CPUS_PER_TRIAL}" -lt 1 ] && CPUS_PER_TRIAL=1

module purge
cd "${CONN2CONN_DIR}"
export RAY_worker_register_timeout_seconds=120 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "Campaign=e0_taskfc Condition=${COND} Source=FC Target=SC Config=${CONFIG} Seed=${SEED} Repo=${CONN2CONN_DIR} Commit=$(git rev-parse --short HEAD) (${NUM_SAMPLES} samples, ${SEARCH_ALG}, ${MAX_CONCURRENT} x ${GPUS_PER_TRIAL} GPU)"

singularity exec --nv \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  --env "RAY_worker_register_timeout_seconds=${RAY_worker_register_timeout_seconds}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1
    unset RAY_TMPDIR
    cd ${CONN2CONN_DIR}
    python main.py --mode prod --model CrossModal_PCA_PLS_learnable --config ${CONFIG} \
      --source FC --target SC --shuffle_seed ${SEED} --data_load_mode precomputed --save_checkpoint \
      --use_tune --search_alg ${SEARCH_ALG} --num_samples ${NUM_SAMPLES} \
      --max_concurrent_trials ${MAX_CONCURRENT} --tune_cpus_per_trial ${CPUS_PER_TRIAL} \
      --tune_gpus_per_trial ${GPUS_PER_TRIAL} --tune_reuse_actors true --report_best_after_tune --store_eval_md
  "
echo "Job Over at $(date)"
