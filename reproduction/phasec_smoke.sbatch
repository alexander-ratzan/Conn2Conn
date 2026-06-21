#!/bin/bash
# PHASE C SMOKE: downstream (cognition + sex leak) on {bv+demo, obs_FC, pred_SC, obs_FC+bv+demo}
# x {CogCryst, sex} x 3 estimators, Glasser seed0 (reuses Glasser/seed0 handoff artifacts),
# then the leak-check verdict. Validates: scalar estimators, lift+paired-permutation,
# residualized score, handoff subject_id join (BP-2), leak guardrail + exempt-flag.
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=grid_phasec
#SBATCH --output=/scratch/ans9868/Conn2Conn/reproduction/slurm-phasec-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
RD=$ROOT/reproduction
cd "$RD"
rm -f DONE_phasec.sentinel ERROR_phasec.sentinel

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
    echo '=== downstream phase-c smoke ===' > phasec_smoke_output.txt
    python run_downstream_grid.py --phasec-smoke >> phasec_smoke_output.txt 2>&1 &&
    echo '=== leak checks ===' >> phasec_smoke_output.txt &&
    python run_leak_checks.py --downstream-csv outputs/phasec_smoke_downstream.csv \
        --out outputs/phasec_smoke_leak_verdict.csv >> phasec_smoke_output.txt 2>&1
" || { echo failed; touch "$RD/ERROR_phasec.sentinel"; exit 1; }
touch "$RD/DONE_phasec.sentinel"
echo done
