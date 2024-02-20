#!/usr/bin/env bash
#SBATCH --job xai-nlp-benchmark
#SBATCH --partition=cpu-7d
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (4G per cpu-core is default)
##SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
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