#!/usr/bin/env bash
#$ -binding linear:8
#$ -N nlp-benchmark-training
##$ -l h_rt=24:00:00
##$ -l s_rt=24:00:00
#$ -l h_vmem=20G
#$ -l mem_free=20G
##$ -m abe
##$ -M rick.wilming@tu-berlin.de

export DATADIR=/home/space/uniml/rick/data/nlp-benchmark
export WORKDIR=/home/space/uniml/rick/$1
cd $WORKDIR
ls -l
apptainer build --fakeroot --force /home/rick/nlp-apptainerfile-training.sif nlp-apptainerfile-training.def
apptainer run --bind $DATADIR:/mnt --bind $WORKDIR:/workdir /home/rick/nlp-apptainerfile-training.sif
