# nlp-benchmark
NLP Benchmark for XAI methods

AD Test Commit.

# Running experiments
1. Generate datasets
```bash
python run_experiments.py --mode=data --config=./configs/gender_project_config.json
```
2. Run model training
```bash
python run_experiments.py --mode=training --config=./artifacts/xai-nlp-benchmark-gender-2024-01-01-01-01-01/data/project_config.json
```

3. Run XAI
```bash
python run_experiments.py --mode=xai --config=./artifacts/xai-nlp-benchmark-gender-2024-01-01-01-01-01/data/project_config.json
```

4. Run Evaluations
```bash
python run_experiments.py --mode=evaluation --config=./artifacts/xai-nlp-benchmark-gender-2024-01-01-01-01-01/data/project_config.json
```

5.  Create visualizations

We have to update `"absolute_dir_to_project"` in `project_config.json` of `"artifacts/nlp-benchmark-.../data/"`:
```
  "visualization": {
    "output_dir": "visualization",
    "absolute_dir_to_project": "local_project_path/xai-nlp-benchmark",
    ..
  }
```

```bash
python run_experiments.py --mode=visualization --config=./artifacts/xai-nlp-benchmark-gender-2024-01-01-01-01-01/data/project_config.json
```

# On hydra

To run the code on the cluster we have to do three steps:
1. Run the data script on your local device and copy it to the cluster
2. Move the code to the cluster
3. Start the cluster script which builds a container and runs it

## Step 1: Copy Data

1. Adjust .env file
Add your hydra username, base_dir and known hosts_file.

Example:
```
HYDRA_BASE_DIR=/home/space/rick
HYDRA_SSH_USER=rick
```

2. Generate datasets
```bash
python run_experiments.py --mode=data --config=./configs/gender_project_config.json
```

A folder is generated using the project name and a timestamp e.g. xai-nlp-benchmark-gender-2024-01-01-01-01-01

3. Adjusting the base_dir
We have to update `"base_dir"` in `project_config.json`:
```
  "general": {
    "seed": 43532791,
    "base_dir": "artifacts",
    "data_scenario": "nlp-benchmark-2023-08-23-15-26-05"
  }
```
If we run experiments locally we set `"base_dir": "artifacts"`, but on the
cluster we use container environments where we mount a data directory to the
running container. Normally, we mount the data directory under `/mnt` which means
we have to update `"base_dir": "artifacts"` to `"base_dir": "/mnt/artifacts"`.


4. Pushing the data to the cluster

We can copy the newly created folder using the copy data script e.g.

```bash
./copy_data_to_cluster.sh xai-nlp-benchmark-gender-2024-01-01-01-01-01
```

## Step 2: Copy Code

Either clone the remote repository or use the `upload_code_to_cluster.py` script.

```bash
python upload_code_to_cluster.py hydra
```

## Step 3. Build & run container

To run the application on the cluster you can use the script available in `./scripts/submit_hydra_jobs.sh`.
Adjust the paths to the root_dir (same as HYDRA_BASE_DIR), config path, code path and mode in the script.

This step is done directly on the cluster e.g. using ssh. 

The machine the code is run on and the timeslot can be configured in `cluster_job_hydra_gpu.sh`.
Further details can be found in the hydra documentation: https://git.tu-berlin.de/ml-group/hydra/documentation

The logger outputs of the container can be found in the code directory under logs.