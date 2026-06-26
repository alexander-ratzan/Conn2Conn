#!/bin/bash
# Mechanism probe: BR vs PLS shrinkage / per-PC recovery (Glasser, seeds 0-9, single job).
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=br_probe
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/br_imputation/logs/probe-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
BRD=$ROOT/reproduction/br_imputation
mkdir -p "$BRD/logs" "$BRD/sentinels" "$BRD/outputs"
cd "$BRD"
rm -f "$BRD/sentinels/DONE_probe.sentinel" "$BRD/sentinels/ERROR_probe.sentinel"
LOG="$BRD/logs/probe.txt"

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
    python probe_shrinkage.py > $LOG 2>&1
" || { echo failed; touch "$BRD/sentinels/ERROR_probe.sentinel"; exit 1; }
touch "$BRD/sentinels/DONE_probe.sentinel"
echo "probe done"
