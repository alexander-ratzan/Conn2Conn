#!/bin/bash
# E2.2 benchmark task (spec v2 E2.2): one model x one seed (array index = shuffle seed), Ray Tune over the model's
# benchmark config, then the best-trial report (its JSON summary in this task's log is what run.py reads).
# Submit through submit.py (it sets --job-name e2_mse_<Model>_<direction>, --time, --array from config.yml):
#   python scripts/experiments/model_benchmark/submit.py --direction sc2fc --models CrossModal_PCA_PLS [--dry-run]
# Positional args: <Model> <direction: sc2fc|fc2sc> <num_samples> <search_alg> <gpus_per_trial> <max_concurrent>
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/%x_%A_%a.err

set -euo pipefail

MODEL="${1:?model}"; DIRECTION="${2:?direction}"; NUM_SAMPLES="${3:?num_samples}"; SEARCH_ALG="${4:?search_alg}"
GPUS_PER_TRIAL="${5:-0.25}"; MAX_CONCURRENT="${6:-4}"
SEED=${SLURM_ARRAY_TASK_ID}
case "${DIRECTION}" in
  sc2fc) SOURCE=SC; TARGET=FC ;;
  fc2sc) SOURCE=FC; TARGET=SC ;;
  *) echo "unknown direction ${DIRECTION}"; exit 1 ;;
esac
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
CONFIG="models/configs/benchmark/mse/${MODEL}.yml"
[ "${DIRECTION}" = "fc2sc" ] && [ -f "${CONN2CONN_DIR}/models/configs/benchmark/mse/${MODEL}_fc2sc.yml" ] && CONFIG="models/configs/benchmark/mse/${MODEL}_fc2sc.yml"
CPUS_PER_TRIAL=$(( ${SLURM_CPUS_PER_TASK:-10} / ${MAX_CONCURRENT} ))
[ "${CPUS_PER_TRIAL}" -lt 1 ] && CPUS_PER_TRIAL=1

module purge
cd "${CONN2CONN_DIR}"
export RAY_worker_register_timeout_seconds=120 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "Campaign=e2_mse Model=${MODEL} Direction=${DIRECTION} Source=${SOURCE} Target=${TARGET} Config=${CONFIG} Seed=${SEED} (${NUM_SAMPLES} samples, ${SEARCH_ALG}, ${MAX_CONCURRENT} x ${GPUS_PER_TRIAL} GPU)"

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
    python main.py \
      --mode prod \
      --model ${MODEL} \
      --config ${CONFIG} \
      --source ${SOURCE} \
      --target ${TARGET} \
      --shuffle_seed ${SEED} \
      --data_load_mode precomputed \
      --save_checkpoint \
      --use_tune \
      --search_alg ${SEARCH_ALG} \
      --num_samples ${NUM_SAMPLES} \
      --max_concurrent_trials ${MAX_CONCURRENT} \
      --tune_cpus_per_trial ${CPUS_PER_TRIAL} \
      --tune_gpus_per_trial ${GPUS_PER_TRIAL} \
      --tune_reuse_actors true \
      --report_best_after_tune \
      --store_eval_md
  "

echo "Job Over at $(date)"
