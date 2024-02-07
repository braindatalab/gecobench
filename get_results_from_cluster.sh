#!/bin/bash
# Script: get_results_from_cluster.sh

# Description: This script copies the results from the cluster to the local machine.

# Arguments:
# $1: The name of the folder containing the results on the cluster. e.g. nlp-benchmark-2024-01-30-21-38-15

# if dot env file exists, source it
if [ -f .env ]; then
  source .env
else
  echo "No .env file found. Please create one."
  exit 1
fi

# If no argument is passed, exit
if [ -z "$1" ]; then
  echo "Data folder name is required. e.g. nlp-benchmark-2024..."
  exit 1
fi

HYDRA_SSH="$HYDRA_SSH_USER@hydra"

scp -r $HYDRA_SSH:$HYDRA_BASE_DIR/data/artifacts/"$1"/data artifacts/$1
scp -r $HYDRA_SSH:$HYDRA_BASE_DIR/data/artifacts/"$1"/training artifacts/$1
scp -r $HYDRA_SSH:$HYDRA_BASE_DIR/data/artifacts/"$1"/xai artifacts/$1
scp -r $HYDRA_SSH:$HYDRA_BASE_DIR/data/artifacts/"$1"/evaluation artifacts/$1

echo "Results copied to artifacts/$1"

