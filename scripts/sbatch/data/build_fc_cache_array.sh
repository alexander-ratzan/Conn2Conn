#!/bin/bash
# Build the compact FC npy caches (Conn2Conn_data/fc/) from the xcp-d combined-run relmats, one array task per
# condition, both parcellations per task (data/data_caching/build_fc_cache.py):
#   0 = rest reproducibility check: rebuild into SLURM_TMPDIR, bitwise-compare to the existing fc/parc-*_hemi-both
#   1-7 = emotion gambling language motor relational social wm -> fc/parc-{P}_hemi-both_task-{T}/ (+ spotcheck)
#   sbatch scripts/sbatch/data/build_fc_cache_array.sh
#   sbatch --array=3 scripts/sbatch/data/build_fc_cache_array.sh          # one condition
# The builder refuses to overwrite an existing cache folder. Afterwards (login node is fine):
#   python data/data_caching/build_fc_cache.py catalog     # fc/catalog.tsv + fc/availability.tsv
# ~0.5 s per subject per worker -> ~2 min per cache folder at 8 workers.
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem=24GB
#SBATCH --job-name=build_fc_cache
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/build_fc_cache_%A_%a.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/build_fc_cache_%A_%a.err
#SBATCH --array=0-7

set -euo pipefail

module purge
CONN2CONN_DIR="/scratch/asr655/neuroinformatics/Conn2Conn"
CACHE_ROOT="/scratch/asr655/neuroinformatics/Conn2Conn_data"
cd "${CONN2CONN_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

CONDITIONS=(rest emotion gambling language motor relational social wm)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
WORKERS=${SLURM_CPUS_PER_TASK:-8}
GIT_COMMIT=$(git -C "${CONN2CONN_DIR}" rev-parse HEAD)
if ! git -C "${CONN2CONN_DIR}" diff --quiet HEAD -- data/data_caching/build_fc_cache.py; then
    GIT_COMMIT="${GIT_COMMIT}+dirty"
fi
if [[ "${CONDITION}" == "rest" ]]; then
    OUT_ROOT="${SLURM_TMPDIR:-/tmp/${USER}_build_fc_cache_${SLURM_JOB_ID}}/rest_check"
else
    OUT_ROOT="${CACHE_ROOT}"
fi

echo "Starting job ${SLURM_JOB_ID} (task ${SLURM_ARRAY_TASK_ID} = ${CONDITION}) on $(hostname) at $(date)"
echo "Builder commit=${GIT_COMMIT}  out_root=${OUT_ROOT}  workers=${WORKERS}"

singularity exec \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  --env "SLURM_JOB_ID=${SLURM_JOB_ID}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    set -euo pipefail
    source /ext3/env.sh
    export PYTHONUNBUFFERED=1
    cd ${CONN2CONN_DIR}
    for PARC in 4S456Parcels Glasser; do
      python data/data_caching/build_fc_cache.py build --condition ${CONDITION} --parcellation \${PARC} \
        --out-root ${OUT_ROOT} --workers ${WORKERS} --git-commit ${GIT_COMMIT}
      if [[ ${CONDITION} == rest ]]; then
        python data/data_caching/build_fc_cache.py compare --a ${OUT_ROOT}/fc/parc-\${PARC}_hemi-both --b ${CACHE_ROOT}/fc/parc-\${PARC}_hemi-both
        cp ${OUT_ROOT}/fc/parc-\${PARC}_hemi-both/manifest.json ${CONN2CONN_DIR}/results/logs/build_fc_cache_${SLURM_JOB_ID}_rest_check_\${PARC}_manifest.json
      else
        python data/data_caching/build_fc_cache.py spotcheck --cache-dir ${CACHE_ROOT}/fc/parc-\${PARC}_hemi-both_task-${CONDITION} \
          --condition ${CONDITION} --parcellation \${PARC} --n 25
      fi
    done
  "

echo "Job Over at $(date)"
