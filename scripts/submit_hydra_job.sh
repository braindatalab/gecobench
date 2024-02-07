#!/bin/bash

# Script Name: submit_hydra_job.sh
#
# Description: Builds and submits a hydra job as specified by the cluster_job_hydra_build.sh, cluster_job_hydra_gpu_run_prebuild.sh scripts
#              Needs to be run on the hydra cluster via ssh. 
#
# Usage:
#   ./submit_hydra_job.sh [build | run | build_and_run]
#
#   - build: Builds the apptainer.sif file by submitting cpu task for 15 minutes. Runs the cluster_job_hydra_build.sh script.
#            The build job is necessary to rerun if the package dependencies have changed.
#   - run: Submits a gpu job. Runs the cluster_job_hydra_gpu_run_prebuild.sh script. Adjust the type of gpu and time limit in the script.
#   - build_and_run: Submits a gpu job, which first builds and then runs the application. Runs the cluster_job_hydra_gpu.sh script.
#                    This has the disadvantage that the build job takes up resources for the gpu job.

ROOT_FOLDER=/home/space/uniml/hjalmar
CODE_FOLDER=xai-nlp-benchmark
CONFIG_PATH=/mnt/artifacts/nlp-benchmark-2024-01-30-21-38-15/data/project_config.json
MODE=training # Mode e.g. training, visualization, evaluation, xai
MAIL=hjalmar.schulz@charite.de

full_code_path=$ROOT_FOLDER/$CODE_FOLDER
cd $full_code_path

if [ "$1" == "build" ]; then
	sbatch --mail-user=$MAIL ./cluster_job_hydra_gpu.sh \
		$ROOT_FOLDER \
		$CODE_FOLDER
elif [ "$1" == "run" ]; then
	sbatch --mail-user=$MAIL ./cluster_job_hydra_gpu_run_prebuild.sh \
		$ROOT_FOLDER \
		$CODE_FOLDER \
		$MODE \
		$CONFIG_PATH
elif [ "$1" == "build_and_run" ]; then
	sbatch --mail-user=$MAIL ./cluster_job_hydra_gpu.sh \
		$ROOT_FOLDER \
		$CODE_FOLDER \
		$MODE \
		$CONFIG_PATH
else
	echo "Invalid mode. Please specify either 'build', 'run' or 'build_and_run'"
	exit 1
fi


