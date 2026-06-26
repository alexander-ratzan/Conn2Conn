#!/bin/bash
# Finalize BR-family (afterok of the array): pool Glasser seeds -> family_auc_br.csv + BR-vs-PLS print.
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=brfam_final
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/br_family/logs/brfam-finalize-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
BFD=$ROOT/reproduction/br_family
mkdir -p "$BFD/logs" "$BFD/sentinels"
cd "$BFD"
rm -f "$BFD/sentinels/DONE_brfam_finalize.sentinel" "$BFD/sentinels/ERROR_brfam_finalize.sentinel"
LOG="$BFD/logs/brfam_finalize.txt"

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export HOME=$ROOT/reproduction/outputs/wandb_home XDG_CACHE_HOME=$ROOT/reproduction/outputs/wandb_cache/xdg
    mkdir -p \$HOME \$XDG_CACHE_HOME
    cd $BFD
    python finalize_br_family.py > $LOG 2>&1
" || { echo failed; touch "$BFD/sentinels/ERROR_brfam_finalize.sentinel"; exit 1; }
touch "$BFD/sentinels/DONE_brfam_finalize.sentinel"
echo "brfam finalize done"
