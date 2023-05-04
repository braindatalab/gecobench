#!/usr/bin/env bash
#$ -binding linear:4
#$ -N nlp-benchmark-training
##$ -l h_rt=24:00:00
##$ -l s_rt=24:00:00
#$ -l h_vmem=20G
#$ -l mem_free=20G
#$ -l cuda=1
##$ -m abe
##$ -M rick.wilming@tu-berlin.de

export DATADIR=/home/space/uniml/rick/data
export WORKDIR=/home/space/uniml/rick/$1
cd $WORKDIR
ls -l
apptainer build --fakeroot --force /home/rick/nlp-apptainerfile.sif nlp-apptainerfile.def
apptainer run --nv --bind $DATADIR:/mnt --bind $WORKDIR:/workdir /home/rick/nlp-apptainerfile.sif "$2" "$3"
