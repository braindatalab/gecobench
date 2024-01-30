#!/usr/bin/env bash
#SBATCH --job xai-nlp-benchmark
#SBATCH --partition=gpu-2h
##SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/xai-nlp-benchmark-job-%j.out

# $1 base_dir e.g. /home/space/uniml/rick
# $2 name of the code folder e.g. xai-nlp-benchmark-2024-01-01-01-01-01
# $3 mode e.g. training, evaluation, xai
# $4 config file

export DATADIR=$1/data # e.g. /home/space/uniml/rick/data
export WORKDIR=$1/$2 # e.g. /home/space/uniml/rick/xai-nlp-benchmark-2024-01-01-01-01-01
cd $WORKDIR
ls -l
apptainer build --fakeroot --force $WORKDIR/nlp-apptainerfile.sif nlp-apptainerfile.def
apptainer run --env "WANDB_API_KEY=$WANDB_API_KEY" --nv --bind $DATADIR:/mnt --bind $WORKDIR:/workdir $WORKDIR/nlp-apptainerfile.sif "$3" "$4"