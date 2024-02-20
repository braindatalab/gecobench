#!/usr/bin/env bash
#SBATCH --job xai-nlp-benchmark
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint="80gb|40gb"
#SBATCH --output=logs/xai-nlp-benchmark-job-%j.out

# $1 PROJECT_DIR
# $2 DATA_DIR
# $3 ARTIFACT_DIR
# $4 mode
# $5 CONFIG_PATH in mounted artifact dir

PROJECT_DIR=$1
DATA_DIR=$2
ARTIFACT_DIR=$3
MODE=$4
CONFIG=$5

cd $PROJECT_DIR
ls -l
apptainer run \
    --env "WANDB_API_KEY=$WANDB_API_KEY" --nv \
    --bind "$DATA_DIR:/mnt/data" \
    --bind "$ARTIFACT_DIR:/mnt/artifacts" \
    --bind "$PROJECT_DIR:/workdir" \
    $PROJECT_DIR/nlp-apptainerfile.sif "$MODE" "$CONFIG"