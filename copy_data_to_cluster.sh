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
cd artifacts
zip -r temp_artifacts.zip $1/data
cd ..

HYDRA_SSH="$HYDRA_SSH_USER@hydra"

echo "Copying data to $HYDRA_SSH_USER@hydra:$HYDRA_BASE_DIR/data/artifacts"

# Create the artifacts folder on the cluster if it doesn't exist
ssh $HYDRA_SSH "mkdir -p $HYDRA_BASE_DIR/data/artifacts"

# Copy the zip file to the cluster
scp artifacts/temp_artifacts.zip $HYDRA_SSH:$HYDRA_BASE_DIR/data/artifacts/temp_artifacts.zip

# Unzip the data
ssh $HYDRA_SSH "unzip $HYDRA_BASE_DIR/data/artifacts/temp_artifacts.zip -d $HYDRA_BASE_DIR/data/artifacts"

# Remove the zip file
ssh $HYDRA_SSH "rm $HYDRA_BASE_DIR/data/artifacts/temp_artifacts.zip"

# Remove the local zip file
rm artifacts/temp_artifacts.zip