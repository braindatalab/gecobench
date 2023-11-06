#!/bin/bash

scp -r rick@hydra:/home/space/uniml/rick/data/artifacts/"$1"/training artifacts/$1
scp -r rick@hydra:/home/space/uniml/rick/data/artifacts/"$1"/xai artifacts/$1
scp -r rick@hydra:/home/space/uniml/rick/data/artifacts/"$1"/evaluation artifacts/$1
