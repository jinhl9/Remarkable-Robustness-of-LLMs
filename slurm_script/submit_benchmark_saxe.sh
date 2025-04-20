#!/bin/bash

# Define variables
#INTERVENTION_TYPES=("swap" "delete")
#INTERVENTION_LAYERS=($(seq 0 2))

INTERVENTION_TYPES=("swap" "delete")
INTERVENTION_LAYERS=($(seq 0 32))
TASKS=("lambada_openai" "arc_easy" "hellaswag")
#BASE_MODEL="openai-community/gpt2-xl"
BASE_MODEL="EleutherAI/pythia-6.9b-deduped"
# Iterate over layers and types
for LAYER in "${INTERVENTION_LAYERS[@]}"; do
    for TYPE in "${INTERVENTION_TYPES[@]}"; do
        for TASK in "${TASKS[@]}"; do
            echo "Running benchmark_saxe.sh for TASK=$TASK, TYPE=$TYPE and LAYER=$LAYER "
            sbatch slurm_script/benchmark_saxe.sbatch "$TYPE" "$LAYER" $BASE_MODEL $TASK
        done
    done
done