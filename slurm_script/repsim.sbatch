#!/bin/bash
#SBATCH --job-name=benchmark_job
#SBATCH --output=slurm_logs_repsim/benchmark_output_%j.log
#SBATCH --error=slurm_logs_repsim/benchmark_error_%j.err
#SBATCH --partition=gpu_saxe
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=100G

module load cuda/11.8
conda activate llm_robustness

# Define variables
BASE_MODEL=$1
LENGTH=$2
echo "Model: $BASE_MODEL LENGTH: $LENGTH"

python representation_similarity.py \
    --model_name $BASE_MODEL \
    --save_dir "/ceph/scratch/jlee/llm-robustness/repsim-layers-cka" \
    --num_samples $LENGTH

echo "Representation similarity for ${BASE_MODEL} completed."