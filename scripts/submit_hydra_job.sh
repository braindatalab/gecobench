#!/bin/bash
# To be run on the hydra cluster

ROOT_FOLDER=/home/space/uniml/hjalmar
CODE_FOLDER=xai-nlp-benchmark-2024-01-01-01-01-01 # Name of the folder containing the code in the root folder
MODE=training # Mode e.g. training, visualization, evaluation, xai
CONFIG_PATH=/mnt/artifacts/nlp-benchmark-2024-01-01-01-01-01/data/project_config.json # path to the project config
MAIL=hjalmar.schulz@charite.de

cd $CODE_FOLDER

sbatch --mail-user=$MAIL ./cluster_job_hydra_gpu.sh \
	$ROOT_FOLDER \
	$CODE_FOLDER \
	$MODE \
	$CONFIG_PATH

