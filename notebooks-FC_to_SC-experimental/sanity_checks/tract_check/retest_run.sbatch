#!/bin/bash
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=retest_pc45
#SBATCH --output=/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/sanity_checks/tract_check/retest-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
TRACT=$ROOT/notebooks-FC_to_SC-experimental/sanity_checks/tract_check
FUR_EXP=$ROOT/notebooks-FC_to_SC-experimental/further_exploration
mkdir -p "$TRACT" "$FUR_EXP"
cd "$TRACT"
rm -f DONE.sentinel ERROR.sentinel
rm -f pc4_pc5_output.txt retest_icc_output.txt

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:rw \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True
    export OMP_NUM_THREADS=16
    export OPENBLAS_NUM_THREADS=16
    export MKL_NUM_THREADS=16
    set -e
    echo '[run] pc4_pc5_confound_localization.py ...'
    cd $FUR_EXP
    time python pc4_pc5_confound_localization.py > $TRACT/pc4_pc5_output.txt 2>&1
    echo '[run] retest_icc_pipeline.py ...'
    cd $TRACT
    time python retest_icc_pipeline.py > $TRACT/retest_icc_output.txt 2>&1
    echo '[run] both finished cleanly.'
" || {
    echo "[run] a script failed; writing ERROR sentinel"
    touch "$TRACT/ERROR.sentinel"
    exit 1
}
touch "$TRACT/DONE.sentinel"
echo "[run] DONE.sentinel written"
