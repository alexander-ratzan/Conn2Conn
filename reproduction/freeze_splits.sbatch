#!/bin/bash
# Freeze the 10 seed splits ONCE (BP-2). Builds 10 seeds x 2 parcellations to cross-verify
# parcellation-independence, writes splits/seed{0..9}.json + SPLITS_MANIFEST.json.
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=grid_freeze
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/slurm-freeze-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
RD=$ROOT/reproduction
cd "$RD"
rm -f DONE_freeze.sentinel ERROR_freeze.sentinel

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
    cd $RD
    python freeze_splits.py > freeze_splits_output.txt 2>&1
" || { echo failed; touch "$RD/ERROR_freeze.sentinel"; exit 1; }
touch "$RD/DONE_freeze.sentinel"
echo done
