#!/bin/bash
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=tract_predict
#SBATCH --output=/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/tractography_predict/slurm-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
TP=$ROOT/notebooks-FC_to_SC-experimental/tractography_predict
TRACT=$ROOT/notebooks-FC_to_SC-experimental/sanity_checks/tract_check
mkdir -p "$TP"
cd "$TP"
rm -f DONE.sentinel ERROR.sentinel
rm -f e1_*_output.txt e2_*_output.txt e3_*_output.txt e4_*_output.txt synth_*_output.txt retest_icc_output.txt

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
    cd $TP
    echo '[run] e1_source_rep_comparison.py ...'
    time python e1_source_rep_comparison.py > e1_source_rep_output.txt 2>&1
    echo '[run] e2_asymmetry_across_reps.py ...'
    time python e2_asymmetry_across_reps.py > e2_asymmetry_output.txt 2>&1
    echo '[run] e3_marginal_r2t.py ...'
    time python e3_marginal_r2t.py > e3_marginal_output.txt 2>&1
    echo '[run] e4_pc3_localization_on_r2t.py ...'
    time python e4_pc3_localization_on_r2t.py > e4_localization_output.txt 2>&1
    echo '[run] synthesize_tractography.py ...'
    time python synthesize_tractography.py > synthesize_tractography_output.txt 2>&1
    echo '[run] retest_icc_pipeline.py (retry after expose_fc_sessions fix) ...'
    cd $TRACT
    time python retest_icc_pipeline.py > retest_icc_output.txt 2>&1
    echo '[run] all finished cleanly.'
" || {
    echo '[run] a script failed; writing ERROR sentinel'
    touch "$TP/ERROR.sentinel"
    exit 1
}
touch "$TP/DONE.sentinel"
echo '[run] DONE.sentinel written'
