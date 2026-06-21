#!/bin/bash
# FULL GRID — one array task per (parcellation x seed) unit = 20 units.
# Each unit: reconstruction (full pairs x 3 estimators) -> handoff artifacts -> downstream
# (full inputs x cognition+leak) -> leak verdict. Per-unit CSVs in outputs/parts/ (no
# collisions); merged + verified + summarized by finalize.sbatch (chained via dependency).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=grid_unit
# RESOURCES (measured): per-unit peak RSS ~10 GB (Glasser) / ~14.4 GB (4S456); CPU eff 68-90%
#   on 8 cores. The first run requested 48G -> only ~30% used -> 4x48=192G tripped
#   QOSMaxMemoryPerUser, throttling to 2-4 concurrent. 24G (66% headroom over 14.4 GB peak)
#   roughly DOUBLES concurrency under the per-user memory QOS. 8 CPU is correctly sized.
# NOTE: NYU submit plugin caps CPU jobs at 4h cluster-wide (5h/8h rejected "CPU job setup
#   not valid"; --exclude is also blocked by policy). 4h covers Glasser (<1.5h) and 4S456
#   (~2-3h, worst 2:48) with margin. Pathological nodes (e.g. cl012) -> cancel+retry.
#SBATCH --array=0-19%4
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/logs/unit-%A_%a.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
RD=$ROOT/reproduction
mkdir -p "$RD/outputs/parts" "$RD/logs" "$RD/sentinels"
cd "$RD"

PARCS=(Glasser 4S456Parcels)
IDX=$SLURM_ARRAY_TASK_ID
PARC=${PARCS[$((IDX/10))]}
SEED=$((IDX%10))
TAG="${PARC}_s${SEED}"
rm -f "$RD/sentinels/DONE_unit_${IDX}.sentinel" "$RD/sentinels/ERROR_unit_${IDX}.sentinel"
LOG="$RD/logs/unit_${TAG}.txt"
echo "unit IDX=$IDX PARC=$PARC SEED=$SEED" > "$LOG"

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
    echo '=== reconstruction ===' >> $LOG
    python run_reconstruction_grid.py --no-wandb --parcellations $PARC --seeds $SEED \
        --csv outputs/parts/recon_${TAG}.csv >> $LOG 2>&1 &&
    echo '=== handoff ===' >> $LOG &&
    python make_handoff_artifacts.py --parcellations $PARC --seeds $SEED >> $LOG 2>&1 &&
    echo '=== downstream ===' >> $LOG &&
    python run_downstream_grid.py --no-wandb --parcellations $PARC --seeds $SEED \
        --csv outputs/parts/down_${TAG}.csv >> $LOG 2>&1 &&
    echo '=== leak (recorded, not unit-killing) ===' >> $LOG &&
    (python run_leak_checks.py --downstream-csv outputs/parts/down_${TAG}.csv \
        --out outputs/parts/leak_${TAG}.csv >> $LOG 2>&1 || echo 'leak verdict recorded' >> $LOG)
" || { echo failed; touch "$RD/sentinels/ERROR_unit_${IDX}.sentinel"; exit 1; }
touch "$RD/sentinels/DONE_unit_${IDX}.sentinel"
echo "unit $IDX done"
