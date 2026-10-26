#!/bin/bash
# Build the timeseries caches (Conn2Conn_data/timeseries/) and the per-run FC caches (Conn2Conn_data/fc/*_dir-*_run-*)
# from the xcp-d per-run TSVs, one array task per condition, both parcellations per task:
#   1. per-run FC (data/data_caching/build_fc_cache.py build --run R) + spotcheck, for each run of the condition
#   2. timeseries (data/data_caching/build_ts_cache.py build; validated against fc/ combined) + spotcheck (rows ==
#      combined TSV, run correlations == the per-run FC from step 1)
#   sbatch scripts/sbatch/data/build_ts_cache_array.sh
#   sbatch --array=1 scripts/sbatch/data/build_ts_cache_array.sh          # one condition
# 0..7 = rest emotion gambling language motor relational social wm. Builders refuse to overwrite existing folders.
# Afterwards (login node is fine): build_fc_cache.py catalog; build_ts_cache.py catalog.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=48GB
#SBATCH --job-name=build_ts_cache
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/build_ts_cache_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/build_ts_cache_%A_%a.err
#SBATCH --array=0-7

set -euo pipefail

module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
cd "${CONN2CONN_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CONDITIONS=(rest emotion gambling language motor relational social wm)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
if [[ "${CONDITION}" == "rest" ]]; then RUNS="LR1 RL1 LR2 RL2"; else RUNS="LR1 RL1"; fi
WORKERS=${SLURM_CPUS_PER_TASK:-8}
GIT_COMMIT=$(git -C "${CONN2CONN_DIR}" rev-parse HEAD)
if ! git -C "${CONN2CONN_DIR}" diff --quiet HEAD -- data/data_caching/; then
    GIT_COMMIT="${GIT_COMMIT}+dirty"
fi

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID} = ${CONDITION}, runs ${RUNS}) on $(hostname) at $(date)"
echo "Builder commit=${GIT_COMMIT}  workers=${WORKERS}"

singularity exec \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1
    cd ${CONN2CONN_DIR}
    for PARC in Glasser 4S456Parcels; do
      for RUN in ${RUNS}; do
        python data/data_caching/build_fc_cache.py build --condition ${CONDITION} --parcellation \${PARC} --run \${RUN} \
          --workers ${WORKERS} --git-commit ${GIT_COMMIT}
        SUFFIX=\$( [[ ${CONDITION} == rest ]] && echo '' || echo _task-${CONDITION} )
        python data/data_caching/build_fc_cache.py spotcheck --condition ${CONDITION} --parcellation \${PARC} --run \${RUN} \
          --cache-dir /scratch/asr655/neuroinformatics/Conn2Conn_data/fc/parc-\${PARC}_hemi-both\${SUFFIX}_dir-\${RUN:0:2}_run-\${RUN:2:1} --n 25
      done
      python data/data_caching/build_ts_cache.py build --condition ${CONDITION} --parcellation \${PARC} \
        --workers ${WORKERS} --git-commit ${GIT_COMMIT}
      python data/data_caching/build_ts_cache.py spotcheck --condition ${CONDITION} --parcellation \${PARC} --n 25
    done
  "

echo "Job Over at $(date)"
