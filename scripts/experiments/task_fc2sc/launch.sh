#!/bin/bash
# Spec v3 E0 task: one parcellation x direction x FC condition x seed (array index = shuffle seed): Ray Tune over
# configs/<parcellation>/<condition>.yml, then the best-trial report (its JSON summary in this task's log is what
# run.py reads). Submit through autopilot.py / submit.py.
# Positional: <parcellation> <direction fc2sc|sc2fc> <condition> <num_samples> <search_alg> <gpus_per_trial> <max_concurrent>
# CONN2CONN_DIR (env) = checkout to run from (default: main). KEEP_RAY=1 keeps this task's Ray temp + Tune storage.
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
PARC="${1:?parcellation}"; DIRECTION="${2:?direction}"; COND="${3:?condition}"
NUM_SAMPLES="${4:?num_samples}"; SEARCH_ALG="${5:?search_alg}"; GPUS_PER_TRIAL="${6:-0.25}"; MAX_CONCURRENT="${7:-4}"
case "${DIRECTION}" in
  fc2sc) SOURCE=FC; TARGET=SC ;;
  sc2fc) SOURCE=SC; TARGET=FC ;;
  *) echo "unknown direction ${DIRECTION}" >&2; exit 1 ;;
esac
SEED=${SLURM_ARRAY_TASK_ID}
CONN2CONN_DIR="${CONN2CONN_DIR:-/scratch/asr655/neuroinformatics/Conn2Conn}"
CONFIG="scripts/experiments/task_fc2sc/configs/${PARC}/${COND}.yml"
CPUS_PER_TRIAL=$(( ${SLURM_CPUS_PER_TASK:-10} / ${MAX_CONCURRENT} )); [ "${CPUS_PER_TRIAL}" -lt 1 ] && CPUS_PER_TRIAL=1
LOG="${CONN2CONN_DIR}/results/logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}.out"

module purge
cd "${CONN2CONN_DIR}"
export RAY_worker_register_timeout_seconds=120 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID}) on $(hostname) at $(date)"
echo "Campaign=e0_taskfc Parcellation=${PARC} Direction=${DIRECTION} Condition=${COND} Source=${SOURCE} Target=${TARGET} Config=${CONFIG} Seed=${SEED} Repo=${CONN2CONN_DIR} Commit=$(git rev-parse --short HEAD) (${NUM_SAMPLES} samples, ${SEARCH_ALG}, ${MAX_CONCURRENT} x ${GPUS_PER_TRIAL} GPU)"

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
    python main.py --mode prod --model CrossModal_PCA_PLS_learnable --config ${CONFIG} --parcellation ${PARC} \
      --source ${SOURCE} --target ${TARGET} --shuffle_seed ${SEED} --data_load_mode precomputed --save_checkpoint \
      --use_tune --search_alg ${SEARCH_ALG} --num_samples ${NUM_SAMPLES} \
      --max_concurrent_trials ${MAX_CONCURRENT} --tune_cpus_per_trial ${CPUS_PER_TRIAL} \
      --tune_gpus_per_trial ${GPUS_PER_TRIAL} --tune_reuse_actors true --report_best_after_tune --store_eval_md
  "

# Ray cleanup (spec v3 backlog: E0.1 left 342 GB). run.py reads only this log, so after a finished best-trial report
# this task's Ray session dir and Tune storage dir are dead weight. Only on success; KEEP_RAY=1 skips it.
if [ "${KEEP_RAY:-0}" != "1" ] && grep -q "Best Tune trial comprehensive summary:" "${LOG}" 2>/dev/null; then
  RUN_NAME=$(grep -m1 -oP "^Tune run: \K\S+" "${LOG}" || true)
  for d in "${CONN2CONN_DIR}/results/ray_tmp/ray_${SLURM_JOB_ID}" \
           ${RUN_NAME:+"${CONN2CONN_DIR}/results/ray_checkpoints/${RUN_NAME}"}; do
    if [ -d "${d}" ]; then echo "Ray cleanup: $(du -sh "${d}" | cut -f1) ${d}"; rm -rf "${d}"; fi
  done
  rm -f "/tmp/ray_${SLURM_JOB_ID}"
fi
echo "Job Over at $(date)"
