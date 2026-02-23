#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=torch_pr_59_tandon_advanced
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=tune_model
#SBATCH --output=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_model_%j.out
#SBATCH --error=/scratch/asr655/neuroinformatics/Conn2Conn/results/logs/tune_model_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=asr655@nyu.edu

module purge
cd /scratch/asr655/neuroinformatics/Conn2Conn/

singularity exec --nv --fakeroot \
  --overlay "/scratch/$USER/envs/kraken_env/overlay-15GB-500K.ext3:ro" \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /bin/bash -lc "
    source /ext3/env.sh
    python main.py --mode prod --model CrossModal_PCA_PLS_learnable --config models/configs/CrossModal_PCA_PLS_learnable.yml --save_checkpoint --use_tune --num_samples 3
  "

echo "Job Over"