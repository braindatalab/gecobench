#!/usr/bin/env bash

ssh rick@cluster-ida "mkdir /home/space/uniml/rick/data/artifacts"
ssh rick@cluster-ida "mkdir /home/space/uniml/rick/data/artifacts/$1"
ssh rick@cluster-ida "mkdir /home/space/uniml/rick/data/artifacts/$1/data"

scp artifacts/$1/data/train_all.pkl rick@cluster-ida:/home/space/uniml/rick/data/artifacts/$1/data/train_all.pkl
scp artifacts/$1/data/train_all.pkl rick@cluster-ida:/home/space/uniml/rick/data/artifacts/$1/data/train_all.pkl
scp artifacts/$1/data/train_subject.pkl rick@cluster-ida:/home/space/uniml/rick/data/artifacts/$1/data/train_subject.pkl
scp artifacts/$1/data/test_female_all.pkl rick@cluster-ida:/home/space/uniml/rick/data/artifacts/$1/data/test_female_all.pkl
scp artifacts/$1/data/test_female_subject.pkl rick@cluster-ida:/home/space/uniml/rick/data/artifacts/$1/data/test_female_subject.pkl
scp artifacts/$1/data/test_male_all.pkl rick@cluster-ida:/home/space/uniml/rick/data/artifacts/$1/data/test_male_all.pkl
scp artifacts/$1/data/test_male_subject.pkl rick@cluster-ida:/home/space/uniml/rick/data/artifacts/$1/data/test_male_subject.pkl
scp artifacts/$1/data/ground_truth_all.pkl rick@cluster-ida:/home/space/uniml/rick/data/artifacts/$1/data/ground_truth_all.pkl
scp artifacts/$1/data/ground_truth_subject.pkl rick@cluster-ida:/home/space/uniml/rick/data/artifacts/$1/data/ground_truth_subject.pkl
scp artifacts/$1/data/project_config.json rick@cluster-ida:/home/space/uniml/rick/data/artifacts/$1/data/project_config.json
