# nlp-benchmark
NLP Benchmark for XAI methods

AD Test Commit.

# Running experiments
1. Generate datasets
```bash
python run_experiments.py --mode=data --config=project_config.json
```
2. Copy data to cluster
```bash
./copy_data_to_cluster.sh
```
3. Run model training via Apptainer
```bash
python run_experiments.py --mode=data --config=project_config.json
```

# On the cluster
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

