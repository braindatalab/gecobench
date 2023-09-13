#!/bin/bash

#SBATCH -o xai-nlp-benchmark.%j.%N.out   # Output-File
##SBATCH -D                   # Working Directory
#SBATCH -J xai-nlp-benchmark          # Job Name
#SBATCH --ntasks=1              # Anzahl Prozesse (CPU-Cores)
#SBATCH --mem=8G              # 500MiB resident memory pro node

##Max Walltime vorgeben:
#SBATCH --time=72:00:00 # Erwartete Laufzeit

#Auf Standard-Knoten rechnen:
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

export DATADIR=/home/users/r/rick/data
export WORKDIR=/home/users/r/rick/$1
cd $WORKDIR
ls -l
module load singularity/3.7.0

ping -c google.com

#singularity build --fakeroot --force /home/rick/nlp-apptainerfile.sif nlp-apptainerfile.def
#singularity run --nv --bind $DATADIR:/mnt --bind $WORKDIR:/workdir /home/users/r/rick/nlp-apptainerfile.sif "$2" "$3"
