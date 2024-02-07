#!/usr/bin/env bash
#SBATCH --job xai-nlp-benchmark
#SBATCH --partition=cpu-7d
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=3G         # memory per cpu-core (4G per cpu-core is default)
##SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/xai-nlp-benchmark-job-%j.out


export DATADIR=/home/space/uniml/rick/data
export WORKDIR=/home/space/uniml/rick/$1
cd $WORKDIR
ls -l
apptainer build --fakeroot --force /home/rick/nlp-apptainerfile.sif nlp-apptainerfile.def
#apptainer run --nv --bind $DATADIR:/mnt --bind $WORKDIR:/workdir /home/rick/nlp-apptainerfile.sif "$2" "$3"
apptainer run --env "WANDB_API_KEY=$WANDB_API_KEY" --bind $DATADIR:/mnt --bind $WORKDIR:/workdir /home/rick/nlp-apptainerfile.sif "$2" "$3"