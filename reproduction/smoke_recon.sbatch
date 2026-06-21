#!/bin/bash
# SMOKE: reconstruction runner on the simple low-dim inputs (bv, bv+demo) -> SC/FC, seed 0,
# both parcellations, pca_pls. Validates data load, low-dim cap, metric panel, frozen-split
# assert, OFFLINE W&B flat logging, and the CSV mirror. Requires frozen splits to exist.
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --job-name=grid_smoke
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/slurm-smoke-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
RD=$ROOT/reproduction
cd "$RD"
rm -f DONE_smoke.sentinel ERROR_smoke.sentinel

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
    export WANDB_MODE=offline WANDB_DIR=$RD/outputs
    cd $RD
    python run_reconstruction_grid.py --smoke > smoke_recon_output.txt 2>&1
" || { echo failed; touch "$RD/ERROR_smoke.sentinel"; exit 1; }
touch "$RD/DONE_smoke.sentinel"
echo done
