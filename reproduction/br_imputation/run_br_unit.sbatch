#!/bin/bash
# BR-only imputation+downstream — one array task per Glasser seed = 10 units (4S456 deferred).
# Each unit: BR-impute pred_* -> save artifacts -> BR-only downstream over the 18-input set ->
# per-seed part CSV. Merged + leak + completeness by finalize_br.sbatch (chained afterok).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=br_unit
#SBATCH --array=0-9%10
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/br_imputation/logs/br-%A_%a.out
# RESOURCES: Glasser-only, BR-only (no KR sweep, no 4S456). Memory driver = PCA(256) on the
#   64,620-dim FC/SC inputs (same as spine, which peaked ~10 GB doing MORE) + cheap 256-dim
#   per-component BR fits -> 16 GB has headroom. 10x16=160 GB < ~192 GB QOSMaxMemoryPerUser, so
#   %10 runs all 10 seeds in one wave. 2 h is generous (spine Glasser unit < 1.5 h doing more).

set -u
ROOT=/scratch/ans9868/Conn2Conn
BRD=$ROOT/reproduction/br_imputation
mkdir -p "$BRD/outputs/parts" "$BRD/logs" "$BRD/sentinels"
cd "$BRD"

PARC=Glasser
SEED=$SLURM_ARRAY_TASK_ID
TAG="${PARC}_s${SEED}"
rm -f "$BRD/sentinels/DONE_br_${SEED}.sentinel" "$BRD/sentinels/ERROR_br_${SEED}.sentinel"
LOG="$BRD/logs/br_${TAG}.txt"
echo "br unit SEED=$SEED PARC=$PARC" > "$LOG"

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    export HOME=$ROOT/reproduction/outputs/wandb_home XDG_CACHE_HOME=$ROOT/reproduction/outputs/wandb_cache/xdg
    mkdir -p \$HOME \$XDG_CACHE_HOME
    cd $BRD
    echo '=== BR impute + downstream ===' >> $LOG &&
    python run_br_unit.py --parc $PARC --seed $SEED >> $LOG 2>&1
" || { echo failed; touch "$BRD/sentinels/ERROR_br_${SEED}.sentinel"; exit 1; }
touch "$BRD/sentinels/DONE_br_${SEED}.sentinel"
echo "br unit $SEED done"
