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

# Set wandb project and entity
export WANDB_PROJECT="cz4042"

# Parse command line arguments
MODEL=$1       # e.g., distilbert
TASK=$2        # e.g., sentiment_sft
PRECISION=$3   # e.g., SFT
ACCELERATOR=$4 # e.g., single_gpu
OPTIONAL_ARGS=$5

echo "START TIME: $(date)"

# Load modules and activate environment
module load cuda/12.2
module load anaconda
source activate alignment-handbook

# Get config file path
CONFIG_FILE=recipes/$MODEL/config_$PRECISION.yaml
echo "Using config file: $CONFIG_FILE"

# Configure and run training
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/accelerate_configs/$ACCELERATOR.yaml \
    scripts/run_$TASK.py $CONFIG_FILE $OPTIONAL_ARGS

echo "END TIME: $(date)"