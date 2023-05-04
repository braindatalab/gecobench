#!/bin/bash

scp -r rick@cluster-ida:/home/space/uniml/rick/"$1"/artifacts/"$2"/training artifacts/$2
scp -r rick@cluster-ida:/home/space/uniml/rick/"$1"/artifacts/"$2"/xai artifacts/$2
scp -r rick@cluster-ida:/home/space/uniml/rick/"$1"/artifacts/"$2"/evaluation artifacts/$2
