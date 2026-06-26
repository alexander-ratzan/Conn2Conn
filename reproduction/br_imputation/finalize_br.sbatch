#!/bin/bash
# Finalize the BR-only run (runs once, afterok of the array): merge parts -> downstream_br.csv,
# leak verdict, expected cells + completeness hard-check. Writes a finalize sentinel.
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=br_finalize
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/br_imputation/logs/br-finalize-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
BRD=$ROOT/reproduction/br_imputation
mkdir -p "$BRD/logs" "$BRD/sentinels"
cd "$BRD"
rm -f "$BRD/sentinels/DONE_br_finalize.sentinel" "$BRD/sentinels/ERROR_br_finalize.sentinel"
LOG="$BRD/logs/br_finalize.txt"

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export HOME=$ROOT/reproduction/outputs/wandb_home XDG_CACHE_HOME=$ROOT/reproduction/outputs/wandb_cache/xdg
    mkdir -p \$HOME \$XDG_CACHE_HOME
    cd $BRD
    python finalize_br.py --seeds 0 1 2 3 4 5 6 7 8 9 > $LOG 2>&1
" || { echo failed; touch "$BRD/sentinels/ERROR_br_finalize.sentinel"; exit 1; }
touch "$BRD/sentinels/DONE_br_finalize.sentinel"
echo "br finalize done"
