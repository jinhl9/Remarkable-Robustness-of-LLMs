#!/bin/bash

# Define variables
#
BASE_MODELS=("EleutherAI/pythia-2.8b-deduped"  "openai-community/gpt2-xl" "EleutherAI/pythia-6.9b" ) 
# Iterate over layers and types 

for BASE_MODEL in "${BASE_MODELS[@]}"; do
    if [[ $BASE_MODEL == *"pythia"* ]]; then
        LENGTH=96
    elif [[ $BASE_MODEL == *"gpt2"* ]]; then
        LENGTH=1024
    fi
    echo "Running repsim.sbatch for MODEL=$BASE_MODEL LENGTH=$LENGTH"
    sbatch slurm_script/repsim.sbatch $BASE_MODEL $LENGTH
done