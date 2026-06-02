#!/bin/bash
#SBATCH --account=torch_pr_60_tandon_priority
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=tractcheck
#SBATCH --output=/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/sanity_checks/tract_check/slurm-%j.out

set -u
TRACT_DIR=/scratch/ans9868/Conn2Conn/notebooks-FC_to_SC-experimental/sanity_checks/tract_check
mkdir -p "$TRACT_DIR"
cd "$TRACT_DIR"
# Clean previous outputs (re-runnable).
rm -f DONE.sentinel ERROR.sentinel
rm -f proxy1_strength_output.txt
rm -f proxy2_distance_output.txt
rm -f decisive_partialled_enrichment_output.txt
rm -f retest_check_note_output.txt

apptainer exec --fakeroot \
  --overlay /scratch/ans9868/kraken_env/unlocked_kraken_env.ext3:rw \
  /share/apps/images/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  bash -lc "
    source /ext3/miniforge3/bin/activate
    conda activate /ext3/miniforge3/envs/kraken_env
    export PYTHONNOUSERSITE=True
    cd $TRACT_DIR
    set -e
    echo '[run_all] proxy1_strength.py ...'
    python proxy1_strength.py > proxy1_strength_output.txt 2>&1
    echo '[run_all] proxy2_distance.py ...'
    python proxy2_distance.py > proxy2_distance_output.txt 2>&1
    echo '[run_all] decisive_partialled_enrichment.py ...'
    python decisive_partialled_enrichment.py > decisive_partialled_enrichment_output.txt 2>&1
    echo '[run_all] retest_check_note.py ...'
    python retest_check_note.py > retest_check_note_output.txt 2>&1
    echo '[run_all] all scripts finished cleanly.'
" || {
    echo "[run_all] one of the scripts failed; writing ERROR sentinel"
    touch "$TRACT_DIR/ERROR.sentinel"
    exit 1
}
touch "$TRACT_DIR/DONE.sentinel"
echo "[run_all] DONE.sentinel written"
