#!/bin/bash
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --job-name=noise_fc
#SBATCH --output=/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check/slurm-%j.out

set -u
ROOT=/scratch/ans9868/Conn2Conn
ND=$ROOT/notebooks-FC_to_SC-experimental/sanity_checks/noise_sanity_check
cd "$ND"
rm -f DONE.sentinel ERROR.sentinel
rm -f *_output.txt

apptainer exec --fakeroot \
  --overlay $ROOT/../kraken_env/unlocked_kraken_env.ext3:ro \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True
    export PYTHONDONTWRITEBYTECODE=1
    export OMP_NUM_THREADS=16
    export OPENBLAS_NUM_THREADS=16
    export MKL_NUM_THREADS=16
    cd $ND
    set -e
    echo '[run] build_fc_cells.py (PREP) ...'
    time python build_fc_cells.py > build_fc_cells_output.txt 2>&1
    echo '[run] a_reliability_ceiling.py ...'
    time python a_reliability_ceiling.py > a_reliability_ceiling_output.txt 2>&1
    echo '[run] b_variance_decomposition.py ...'
    time python b_variance_decomposition.py > b_variance_decomposition_output.txt 2>&1
    echo '[run] f_discriminability.py ...'
    time python f_discriminability.py > f_discriminability_output.txt 2>&1
    echo '[run] e_crossmodal_disattenuation.py ...'
    time python e_crossmodal_disattenuation.py > e_crossmodal_disattenuation_output.txt 2>&1
    echo '[run] synthesize_noise.py ...'
    time python synthesize_noise.py > synthesize_noise_output.txt 2>&1
    echo '[run] all finished cleanly.'
" || { echo '[run] a script failed'; touch "$ND/ERROR.sentinel"; exit 1; }
touch "$ND/DONE.sentinel"
echo '[run] DONE.sentinel written'
