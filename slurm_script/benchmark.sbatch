#!/bin/bash
#SBATCH --job-name=benchmark_job
#SBATCH --output=slurm_logs_a100/benchmark_output_%j.log
#SBATCH --error=slurm_logs_a100/benchmark_error_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G

module load cuda/11.8
conda activate llm_robustness

# Define variables
BASE_MODEL=$3
INTERVENTION_TYPE=$1
LAYER_NUMBER=$2
TASK=$4
MODEL_SAVE_PATH="/ceph/scratch/jlee/llm-robustness/temp_models/${BASE_MODEL}/${TASK}"

echo "Intervention Type: $INTERVENTION_TYPE"
echo "Layer Number: $LAYER_NUMBER"
mkdir -p $MODEL_SAVE_PATH
OUTPUT_PATH="/ceph/scratch/jlee/llm-robustness/benchmark_results/${TASK}/${BASE_MODEL}"
# Define arrays for intervention types and layer numbers

if [ -d "${OUTPUT_PATH}/${INTERVENTION_TYPE}/layer_${LAYER_NUMBER}" ]; then
    echo "Directory ${OUTPUT_PATH}/${INTERVENTION_TYPE}/layer_${LAYER_NUMBER} exists. Exiting..."
    exit 1
else
    echo "Directory ${OUTPUT_PATH}/${INTERVENTION_TYPE}/layer_${LAYER_NUMBER} does not exist. Proceeding..."
fi

python benchmark.py \
    --model_name $BASE_MODEL \
    --save_basedir $MODEL_SAVE_PATH \
    --intervention_type $INTERVENTION_TYPE \
    --intervention_layer $LAYER_NUMBER

# Step 2: Evaluate the saved model using lm_eval
lm_eval --model hf \
    --model_args "pretrained=${MODEL_SAVE_PATH}/${INTERVENTION_TYPE}/layer_${LAYER_NUMBER}" \
    --tasks $TASK \
    --batch_size 'auto' \
    --output_path "${OUTPUT_PATH}/${INTERVENTION_TYPE}/layer_${LAYER_NUMBER}"
echo "Evaluation completed for ${INTERVENTION_TYPE} at layer ${LAYER_NUMBER}"

# Step 3: Delete the saved model
rm -rf $MODEL_SAVE_PATH/$INTERVENTION_TYPE/layer_$LAYER_NUMBER
echo "Deleted the saved model at ${MODEL_SAVE_PATH}/${INTERVENTION_TYPE}/layer_${LAYER_NUMBER}"
