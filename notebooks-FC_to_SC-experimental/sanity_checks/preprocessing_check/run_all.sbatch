#!/bin/bash
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --job-name=preprocheck
#SBATCH --output=/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check/slurm-%j.out

set -u
PREP_DIR=/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/sanity_checks/preprocessing_check
mkdir -p "$PREP_DIR"
cd "$PREP_DIR"
rm -f DONE.sentinel ERROR.sentinel
rm -f method_a_pca_pls_pca_output.txt method_b_full_pls_output.txt
rm -f method_c_jl_pls_pca_output.txt synthesize_reduction_axis_output.txt

apptainer exec --fakeroot \
  --overlay /scratch/ans9868/kraken_env/unlocked_kraken_env.ext3:rw \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True
    export OMP_NUM_THREADS=32
    export OPENBLAS_NUM_THREADS=32
    export MKL_NUM_THREADS=32
    cd $PREP_DIR
    set -e
    echo '[run_all] method_a_pca_pls_pca.py ...'
    time python method_a_pca_pls_pca.py > method_a_pca_pls_pca_output.txt 2>&1
    echo '[run_all] method_b_full_pls.py ...'
    time python method_b_full_pls.py > method_b_full_pls_output.txt 2>&1
    echo '[run_all] method_c_jl_pls_pca.py ...'
    time python method_c_jl_pls_pca.py > method_c_jl_pls_pca_output.txt 2>&1
    echo '[run_all] synthesize_reduction_axis.py ...'
    time python synthesize_reduction_axis.py > synthesize_reduction_axis_output.txt 2>&1
    echo '[run_all] all scripts finished cleanly.'
" || {
    echo "[run_all] one script failed; writing ERROR sentinel"
    touch "$PREP_DIR/ERROR.sentinel"
    exit 1
}
touch "$PREP_DIR/DONE.sentinel"
echo "[run_all] DONE.sentinel written"
