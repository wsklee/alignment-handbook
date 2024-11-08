#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -x -e

# Parse command line arguments
MODEL=$1       # e.g., distilbert, zephyr-7b-beta
TASK=$2        # e.g., sft, dpo, sentiment_sft
PRECISION=$3   # e.g., full, qlora, SFT
ACCELERATOR=$4 # e.g., single_gpu, deepspeed_zero3
OPTIONAL_ARGS=$5

echo "START TIME: $(date)"

# Load modules and activate environment
module load cuda/12.1
module load anaconda
source activate alignment-handbook

# Set cache directories
export TRANSFORMERS_CACHE="/path/to/your/cache/directory"
export HF_HOME="/path/to/your/cache/directory/huggingface"

# Get config file path
CONFIG_FILE=recipes/$MODEL/config_$PRECISION.yaml

# Parse gradient accumulation steps from config
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')

# If gradient_accumulation_steps is provided in optional args, use that instead
if [[ ! -z "$OPTIONAL_ARGS" ]]; then
    IFS=' ' read -ra ARGS <<< "$OPTIONAL_ARGS"
    for arg in "${ARGS[@]}"; do
        if [[ "$arg" == "--gradient_accumulation_steps="* ]]; then
            GRAD_ACC_STEPS="${arg#*=}"
            break
        fi
    done
fi

echo "Using config file: $CONFIG_FILE"
echo "Gradient accumulation steps: $GRAD_ACC_STEPS"

# Configure training command
export CMD="scripts/run_$TASK.py $CONFIG_FILE $OPTIONAL_ARGS"

# Configure launcher with appropriate settings
export LAUNCHER="ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/$ACCELERATOR.yaml \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    "

# Run the training
echo "Running command: $LAUNCHER $CMD"
$LAUNCHER $CMD

echo "END TIME: $(date)"