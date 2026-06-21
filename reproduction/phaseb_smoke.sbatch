#!/bin/bash
# PHASE B SMOKE: reconstruction slice (FC<->SC asymmetry, connectome+bv+demo block BP-1,
# FC->FC oracle) x all 3 estimators (pca_pls, bayesian_ridge, kernel_ridge 3x3), Glasser seed0
# + handoff artifacts (pred_*, recon_per_subject, split_index) for Glasser seed0.
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=grid_phaseb
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/slurm-phaseb-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
RD=$ROOT/reproduction
cd "$RD"
rm -f DONE_phaseb.sentinel ERROR_phaseb.sentinel

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
    export WANDB_MODE=offline WANDB_DIR=$RD/outputs
    export HOME=$RD/outputs/wandb_home XDG_CACHE_HOME=$RD/outputs/wandb_cache/xdg
    mkdir -p \$HOME \$XDG_CACHE_HOME
    cd $RD
    echo '=== reconstruction phase-b smoke ===' > phaseb_smoke_output.txt
    python run_reconstruction_grid.py --phaseb-smoke >> phaseb_smoke_output.txt 2>&1 &&
    echo '=== handoff artifacts (Glasser seed0) ===' >> phaseb_smoke_output.txt &&
    python make_handoff_artifacts.py --parcellations Glasser --seeds 0 >> phaseb_smoke_output.txt 2>&1
" || { echo failed; touch "$RD/ERROR_phaseb.sentinel"; exit 1; }
touch "$RD/DONE_phaseb.sentinel"
echo done
