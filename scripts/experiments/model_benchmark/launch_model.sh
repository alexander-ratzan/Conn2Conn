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
# Campaign settings from submit.py / autopilot.py (--export): E2.2 defaults; spec v3 E3 sets the composite ones.
TUNE_REUSE_ACTORS="${TUNE_REUSE_ACTORS:-true}"; TUNE_LIGHTNING_CHECKPOINTS="${TUNE_LIGHTNING_CHECKPOINTS:-false}"
CAMPAIGN="${CAMPAIGN:-e2_mse}"; CONFIG_DIR="${CONFIG_DIR:-models/configs/benchmark/mse}"; REPORT_METRICS="${REPORT_METRICS:-val_demeaned_r}"
CONFIG="${CONFIG_DIR}/${MODEL}.yml"
[ "${DIRECTION}" = "fc2sc" ] && [ -f "${CONN2CONN_DIR}/${CONFIG_DIR}/${MODEL}_fc2sc.yml" ] && CONFIG="${CONFIG_DIR}/${MODEL}_fc2sc.yml"
CPUS_PER_TRIAL=$(( ${SLURM_CPUS_PER_TASK:-10} / ${MAX_CONCURRENT} ))
[ "${CPUS_PER_TRIAL}" -lt 1 ] && CPUS_PER_TRIAL=1

module purge
cd "${CONN2CONN_DIR}"
export RAY_worker_register_timeout_seconds=120 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "Campaign=${CAMPAIGN} Model=${MODEL} Direction=${DIRECTION} Source=${SOURCE} Target=${TARGET} Config=${CONFIG} Seed=${SEED} (${NUM_SAMPLES} samples, ${SEARCH_ALG}, ${MAX_CONCURRENT} x ${GPUS_PER_TRIAL} GPU, reuse_actors=${TUNE_REUSE_ACTORS}, lightning_ckpt=${TUNE_LIGHTNING_CHECKPOINTS})"

# GPU utilisation log (the cluster cancels jobs averaging < 50% GPU use for > 2 h, warns below 70%): one sample per
# 30 s next to this task's log (<log>.gpu.csv); the mean is printed when the task exits.
GPU_LOG="${CONN2CONN_DIR}/results/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}.gpu.csv"
GPU_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -l 30 > "${GPU_LOG}" 2>/dev/null &
  GPU_PID=$!
fi
gpu_summary() {
  if [ -n "${GPU_PID}" ]; then kill "${GPU_PID}" 2>/dev/null || true; fi
  if [ -s "${GPU_LOG}" ]; then
    awk -F', *' '{s += $2; n++} END {if (n) printf "GPU util: mean %.1f%% over %d samples (30 s)\n", s / n, n}' "${GPU_LOG}"
  fi
}
trap gpu_summary EXIT

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
      --tune_reuse_actors ${TUNE_REUSE_ACTORS:-true} \
      --tune_lightning_checkpoints ${TUNE_LIGHTNING_CHECKPOINTS} \
      --report_best_after_tune \
      --report_metrics ${REPORT_METRICS//+/,} \
      --store_eval_md
  "

# Ray cleanup (spec v3 backlog; as task_bidirectional_bench/launch.sh): after a finished best-trial report this task's
# Ray session dir and Tune storage dir are dead weight (run.py reads only this log). Only on success; KEEP_RAY=1 skips.
LOG="${CONN2CONN_DIR}/results/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}.out"
if [ "${KEEP_RAY:-0}" != "1" ] && grep -q "Best Tune trial comprehensive summary:" "${LOG}" 2>/dev/null; then
  RUN_NAME=$(grep -m1 -oP "^Tune run: \K\S+" "${LOG}" || true)
  for d in "${CONN2CONN_DIR}/results/ray_tmp/ray_${SLURM_JOB_ID}" \
           ${RUN_NAME:+"${CONN2CONN_DIR}/results/ray_checkpoints/${RUN_NAME}"}; do
    # non-fatal: a W&B logging actor can still be writing into a trial dir (2026-10-21: "Directory not empty")
    if [ -d "${d}" ]; then echo "Ray cleanup: $(du -sh "${d}" | cut -f1) ${d}"; rm -rf "${d}" || echo "Ray cleanup incomplete: ${d}"; fi
  done
  rm -f "/tmp/ray_${SLURM_JOB_ID}"
fi
echo "Job Over at $(date)"
