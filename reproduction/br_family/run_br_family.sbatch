#!/bin/bash
# BR-family (F6/F7 heritability) — one array task per Glasser seed = 10 units (4S456 deferred).
# Each unit builds the 8 family variants with BR imputation, extracts per-pair sims, saves npz.
# Pooled aggregation runs in finalize_br_family.sbatch (chained afterok).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=brfam_unit
#SBATCH --array=0-9%10
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/br_family/logs/brfam-%A_%a.out
# RESOURCES: BR variant build = 4 BR imputations + 1 combined_predict (BR) per seed on the
#   64,620-dim Glasser inputs. Memory driver is the PCA(256) (~10 GB on spine) -> 16 GB ample.
#   4h limit (vs the 2h that timed out a slow-node straggler in the cognition run) gives margin.

set -u
ROOT=/scratch/ans9868/Conn2Conn
BFD=$ROOT/reproduction/br_family
mkdir -p "$BFD/outputs/parts" "$BFD/logs" "$BFD/sentinels"
cd "$BFD"

SEED=$SLURM_ARRAY_TASK_ID
TAG="Glasser_s${SEED}"
rm -f "$BFD/sentinels/DONE_brfam_${SEED}.sentinel" "$BFD/sentinels/ERROR_brfam_${SEED}.sentinel"
LOG="$BFD/logs/brfam_${TAG}.txt"
echo "brfam unit SEED=$SEED PARC=Glasser" > "$LOG"

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
    cd $BFD
    echo '=== BR family variants + pair sims ===' >> $LOG &&
    python run_br_family.py --parc Glasser --seed $SEED >> $LOG 2>&1
" || { echo failed; touch "$BFD/sentinels/ERROR_brfam_${SEED}.sentinel"; exit 1; }
touch "$BFD/sentinels/DONE_brfam_${SEED}.sentinel"
echo "brfam unit $SEED done"
