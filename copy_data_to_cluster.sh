#!/usr/bin/env bash

ssh rick@hydra "mkdir /home/space/uniml/rick/data/artifacts"
ssh rick@hydra "mkdir /home/space/uniml/rick/data/artifacts/$1"
ssh rick@hydra "mkdir /home/space/uniml/rick/data/artifacts/$1/data"

scp artifacts/$1/data/training_all.pkl rick@hydra:/home/space/uniml/rick/data/artifacts/$1/data/training_all.pkl
scp artifacts/$1/data/training_subject.pkl rick@hydra:/home/space/uniml/rick/data/artifacts/$1/data/training_subject.pkl
scp artifacts/$1/data/test_female_all.pkl rick@hydra:/home/space/uniml/rick/data/artifacts/$1/data/test_female_all.pkl
scp artifacts/$1/data/test_female_subject.pkl rick@hydra:/home/space/uniml/rick/data/artifacts/$1/data/test_female_subject.pkl
scp artifacts/$1/data/test_subject.pkl rick@hydra:/home/space/uniml/rick/data/artifacts/$1/data/test_subject.pkl
scp artifacts/$1/data/test_all.pkl rick@hydra:/home/space/uniml/rick/data/artifacts/$1/data/test_all.pkl
scp artifacts/$1/data/test_male_all.pkl rick@hydra:/home/space/uniml/rick/data/artifacts/$1/data/test_male_all.pkl
scp artifacts/$1/data/test_male_subject.pkl rick@hydra:/home/space/uniml/rick/data/artifacts/$1/data/test_male_subject.pkl
scp artifacts/$1/data/project_config.json rick@hydra:/home/space/uniml/rick/data/artifacts/$1/data/project_config.json
