#!/usr/bin/env bash

ssh hpc "mkdir /home/users/r/rick/data/artifacts"
ssh hpc "mkdir /home/users/r/rick/data/artifacts/$1"
ssh hpc "mkdir /home/users/r/rick/data/artifacts/$1/data"

scp artifacts/$1/data/training_all.pkl rick@hpc:/home/users/r/rick/data/artifacts/$1/data/training_all.pkl
scp artifacts/$1/data/training_subject.pkl rick@hpc:/home/users/r/rick/data/artifacts/$1/data/training_subject.pkl
scp artifacts/$1/data/test_female_all.pkl rick@hpc:/home/users/r/rick/data/artifacts/$1/data/test_female_all.pkl
scp artifacts/$1/data/test_female_subject.pkl rick@hpc:/home/users/r/rick/data/artifacts/$1/data/test_female_subject.pkl
scp artifacts/$1/data/test_subject.pkl rick@hpc:/home/users/r/rick/data/artifacts/$1/data/test_subject.pkl
scp artifacts/$1/data/test_all.pkl rick@hpc:/home/users/r/rick/data/artifacts/$1/data/test_all.pkl
scp artifacts/$1/data/test_male_all.pkl rick@hpc:/home/users/r/rick/data/artifacts/$1/data/test_male_all.pkl
scp artifacts/$1/data/test_male_subject.pkl rick@hpc:/home/users/r/rick/data/artifacts/$1/data/test_male_subject.pkl
scp artifacts/$1/data/project_config.json rick@hpc:/home/users/r/rick/data/artifacts/$1/data/project_config.json
