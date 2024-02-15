#!/usr/bin/env bash
#SBATCH --job xai-nlp-benchmark
#SBATCH --partition=cpu-test
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/xai-nlp-benchmark-job-%j.out

PROJECT_DIR=$1 # e.g. /home/space/uniml/rick/xai-nlp-benchmark-2024-01-01-01-01-01
cd $PROJECT_DIR
ls -l
apptainer build --fakeroot --force $PROJECT_DIR/nlp-apptainerfile.sif nlp-apptainerfile.def