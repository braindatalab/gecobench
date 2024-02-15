#!/usr/bin/env bash

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

# ZIP the data
# path e.g. nlp-benchmark_2024-02-15-09-58-06
pushd artifacts/data
zip -r temp_artifacts.zip $1/*
popd

HYDRA_SSH="$HYDRA_SSH_USER@hydra"

echo "Copying data to $HYDRA_SSH_USER@hydra:$HYDRA_DATA_DIR"

# Copy the zip file to the cluster
scp artifacts/data/temp_artifacts.zip $HYDRA_SSH:$HYDRA_DATA_DIR/temp_artifacts.zip

# Unzip the data
ssh $HYDRA_SSH "unzip $HYDRA_DATA_DIR/temp_artifacts.zip -d $HYDRA_DATA_DIR"

# Remove the zip file
ssh $HYDRA_SSH "rm $HYDRA_DATA_DIR/temp_artifacts.zip"

# Remove the local zip file
rm artifacts/data/temp_artifacts.zip