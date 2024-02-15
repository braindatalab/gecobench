# nlp-benchmark

NLP Benchmark for XAI methods

# Building the datasets

Currently we have four datasets: `gender_all`, `gender_subj`, `sentiment_twitter` and `sentiment_imdb`.
The config in `config/dataset_config.json` specifies the datasets and the parameters for the data generation.

To generate the datasets we can run the following command:

```bash
python generate_data.py --config=./configs/dataset_config.json
```

This will generate a timestamped folder in the `artifacts/data` directory locally.
To upload the data to the cluster we can use the `copy_data_to_cluster.sh` script. This requires the environment variables to be set in the `.env` file as described below.

```bash
./scripts/hydra/copy_data_to_cluster.sh nlp-benchmark_2024-02-15-10-14-37
```

Lastly, the project config has to be updated to point to the correct data directory in `configs/gender_project_config.json`:

```json
{
  ...
  "data" {
    "data_dir": "/path/to/nlp-benchmark_2024-02-15-10-14-37"
  },
  ...
}
```

# Running experiments locally

1. Setup a experiment run
   This creates a timestamped folder for the artifacts and copies the config files to the folder.

```bash
python setup_experiment.py
```

The output gives you the instructions to run the different steps of the experiment.

2. Run model experiment

Set the mode to `training`, `xai`, `evaluation` or `visualization` and the config to the project config. The modes depend on each other and have to be run in the order `training`, `xai`, `evaluation` and `visualization`.

```bash
python run_experiments.py --mode=MODE --config=artifacts/xai-nlp-benchmark-2024-02-15-16-45-19/configs/gender_project_config.json
```

# On hydra

To run the code on the cluster we have to do three steps:

## Step 1: Setup environment

Copy the `.env.example` file to `.env` and fill in the environment variables.
The script assumes you have added lazy access to hydra in your ssh config, as described in the hydra documentation.
https://git.tu-berlin.de/ml-group/hydra/documentation

```
HYDRA_SSH_USER=hjalmar
HYDRA_DATA_DIR=/home/space/datasets/xai_nlp # The path where to place the data on the cluster
HYDRA_PROJECT_DIR=/home/space/uniml/hjalmar/xai-nlp-benchmark # The path to the code on the cluster
KNOWN_HOSTS=/home/hjall/.ssh/known_hosts
```

## Step 2: Move the code to the cluster

Either clone the remote repository (recommended) or use the `upload_code_to_cluster.py` script.

```bash
python ./scripts/hydra/upload_code_to_cluster.py hydra
```

## Step 4. Setup the project

Ssh into the cluster and navigate to the code directory.
As mentioned above, this creates a timestamped folder for the artifacts and copies the config files to the folder.

```bash
python setup_experiment.py
```

By default it will create the artifacts folder in the code directory.

## Step 5. Build and run the container

To run the code we need to first build the container. This step only needs to be repeated if the dependencies change.

```bash
python ./scripts/hydra/submit_hydra_job.py --mode build --config ./artifacts/xai-nlp-benchmark-2024-02-15-16-45-19/configs/sentiment_project_config.json
```

Afterwards we can run the container with the following command:

```bash
python ./scripts/hydra/submit_hydra_job.py --mode training --config ./artifacts/xai-nlp-benchmark-2024-02-15-16-45-19/configs/sentiment_project_config.json
```

Again, the mode depends on the previous steps and has to be run in the order `training`, `xai`, `evaluation` and `visualization`.

The machine the code is run on and the timeslot can be configured in `./scripts/hydra/cluster_job_hydra_gpu.sh`.
Further details can be found in the hydra documentation: https://git.tu-berlin.de/ml-group/hydra/documentation
The logger outputs of the container can be found in the code directory under logs.

## Step 4. View and cancel jobs

To view your current jobs: run e.g. `squeue --user=hjalmar`
To cancel a job run `scancel job_id` with the job id you get from the command above.

## Step 5. Retrieve results

To copy the results from the cluster to your local machine you can use the `get_results_from_cluster.sh` script.

```bash
./scripts/hydra/get_results_from_cluster.sh xai-nlp-benchmark-2024...
```
