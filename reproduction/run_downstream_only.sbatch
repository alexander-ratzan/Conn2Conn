#!/bin/bash
# DOWNSTREAM-ONLY re-run for a single (parc, seed) whose reconstruction + handoff artifacts
# are ALREADY saved (recon part CSV complete + outputs/artifacts/.../pred_* present). Skips the
# expensive recon redo — used to recover a unit that timed out during the downstream stage.
# Usage: sbatch run_downstream_only.sbatch <PARC> <SEED>
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=grid_dsonly
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/logs/dsonly-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
RD=$ROOT/reproduction
cd "$RD"
mkdir -p logs sentinels
PARC=$1
SEED=$2
TAG="${PARC}_s${SEED}"
rm -f "$RD/sentinels/DONE_dsonly_${TAG}.sentinel" "$RD/sentinels/ERROR_dsonly_${TAG}.sentinel"
LOG="$RD/logs/dsonly_${TAG}.txt"
echo "downstream-only PARC=$PARC SEED=$SEED" > "$LOG"

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    export HOME=$RD/outputs/wandb_home XDG_CACHE_HOME=$RD/outputs/wandb_cache/xdg
    mkdir -p \$HOME \$XDG_CACHE_HOME
    cd $RD
    echo '=== downstream (recon+handoff reused from disk) ===' >> $LOG
    python run_downstream_grid.py --no-wandb --parcellations $PARC --seeds $SEED \
        --csv outputs/parts/down_${TAG}.csv >> $LOG 2>&1 &&
    echo '=== leak ===' >> $LOG &&
    (python run_leak_checks.py --downstream-csv outputs/parts/down_${TAG}.csv \
        --out outputs/parts/leak_${TAG}.csv >> $LOG 2>&1 || echo 'leak recorded' >> $LOG)
" || { echo failed; touch "$RD/sentinels/ERROR_dsonly_${TAG}.sentinel"; exit 1; }
touch "$RD/sentinels/DONE_dsonly_${TAG}.sentinel"
echo "downstream-only $TAG done"
