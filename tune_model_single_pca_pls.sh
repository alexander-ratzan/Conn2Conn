#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=run_model_single
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/run_model_single_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/run_model_single_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=asr655@nyu.edu

module purge
cd /scratch/asr655/neuroinformatics/Conn2Conn/
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "${RAY_TMPDIR}"

# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1

export TUNE_CPUS_PER_TRIAL=4
export TUNE_GPUS_PER_TRIAL=1
export MAX_CONCURRENT_TRIALS=1

singularity exec --nv \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  --env "RAY_TMPDIR=${RAY_TMPDIR}" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    source /ext3/env.sh
    python main.py \
      --mode prod \
      --model CrossModal_PCA_PLS \
      --config models/configs/CrossModal_PCA_PLS.yml \
      --source SC \
      --target FC \
      --save_checkpoint \
      --use_tune \
      --num_samples 1 \
      --max_concurrent_trials ${MAX_CONCURRENT_TRIALS} \
      --tune_cpus_per_trial ${TUNE_CPUS_PER_TRIAL} \
      --tune_gpus_per_trial ${TUNE_GPUS_PER_TRIAL} \
      --report_best_after_tune \
      --store_eval_md
  "

echo "Job Over"
