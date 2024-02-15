# Creates a new folder for the experiment and copies the necessary files to run the experiment

import os
import argparse
from datetime import datetime
import json

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def get_git_commit_hash():
    try:
        return os.popen("git rev-parse HEAD").read().strip()
    except:
        return "Unknown"


def main():
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = argparse.ArgumentParser(description='Setup an experiment')
    parser.add_argument(
        '--experiment_name',
        type=str,
        help='Name of the experiment',
        default="xai-nlp-benchmark-" + date,
        required=False,
    )
    parser.add_argument(
        '--artifact_path',
        type=str,
        help='Path to the artifact',
        default="./artifacts",
        required=False,
    )
    args = parser.parse_args()

    # Create the experiment folder
    artifacts_dir = os.path.join(args.artifact_path, args.experiment_name)
    os.makedirs(artifacts_dir, exist_ok=True)
    config_dir = os.path.join(artifacts_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)

    print("Created experiment folder", artifacts_dir, end="\n\n")

    # Update paths in the config files
    project_configs = filter(
        lambda x: x.endswith("project_config.json"), os.listdir("configs")
    )

    git_commit_hash = get_git_commit_hash()
    for config in project_configs:
        with open(f"./configs/{config}", "r") as f:
            project_config = json.load(f)
            project_config["general"]["artifacts_dir"] = os.path.abspath(artifacts_dir)
            project_config["general"]["project_dir"] = os.path.abspath(".")
            project_config["general"]["git_commit"] = git_commit_hash

        config_path = os.path.join(config_dir, config)
        with open(config_path, "w") as f:
            json.dump(project_config, f, indent=4)

        print(f"{color.BOLD}Updated config: {config} {color.END}")
        print("Run the experiment using with e.g.:")
        print(
            f"{color.GREEN}python run_experiments.py --config {config_path} --mode training{color.END}\n"
        )
        print("or to run it on the cluster:")
        print(f"{color.GREEN}python3 ./scripts/submit_hydra_job.py --mode build --config {config_path}{color.END}")
        print(f"{color.GREEN}python3 ./scripts/submit_hydra_job.py --mode training --config {config_path}{color.END}\n")

if __name__ == "__main__":
    main()
