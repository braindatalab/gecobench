#!/bin/bash

# Script Name: submit_hydra_job.sh
#
# Description: Builds and submits a hydra job as specified by the cluster_job_hydra_build.sh, cluster_job_hydra_gpu_run_prebuild.sh scripts
#              Needs to be run on the hydra cluster via ssh. 
#
# Usage:
#   ./submit_hydra_job.sh [build | run | build_and_run]
#

# full_code_path=$ROOT_FOLDER/$CODE_FOLDER
# cd $full_code_path

# if [ "$1" == "build" ]; then
# 	sbatch --mail-user=$MAIL ./cluster_job_hydra_gpu.sh \
# 		$ROOT_FOLDER \
# 		$CODE_FOLDER
# elif [ "$1" == "run" ]; then
# 	sbatch --mail-user=$MAIL ./cluster_job_hydra_gpu_run_prebuild.sh \
# 		$ROOT_FOLDER \
# 		$CODE_FOLDER \
# 		$MODE \
# 		$CONFIG_PATH
# elif [ "$1" == "build_and_run" ]; then
# 	sbatch --mail-user=$MAIL ./cluster_job_hydra_gpu.sh \
# 		$ROOT_FOLDER \
# 		$CODE_FOLDER \
# 		$MODE \
# 		$CONFIG_PATH
# else
# 	echo "Invalid mode. Please specify either 'build', 'run' or 'build_and_run'"
# 	exit 1
# fi


