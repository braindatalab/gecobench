import argparse
import os
import sys
import json


def main():
    parser = argparse.ArgumentParser(description='Submit a hydra job')
    parser.add_argument(
        '--config',
        dest='config',
        required=True,
        help='File path to project config.',
        type=str,
        default=1,
    )

    parser.add_argument(
        '--mode',
        dest='mode',
        required=True,
        help='Mode of the hydra job. build, training, visualization, evaluation, xai',
        type=str,
        default=1,
    )

    parser.add_argument(
        "--device",
        dest="device",
        required=False,
        help="Device: gpu or cpu",
        type=str,
        default="gpu",
    )

    parser.add_argument(
        "--mail",
        dest="mail",
        required=False,
        help="Email",
        type=str,
        default="hjalmar.schulz@charite.de",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        project_config = json.load(f)
        data_dir = project_config["data"]["data_dir"]
        artifact_dir = project_config["general"]["artifacts_dir"]
        project_dir = project_config["general"]["project_dir"]

    if args.mode == "build":
        os.system(
            f"sbatch --mail-user={args.mail} ./hydra/clust_job_hydra_build.sh {project_dir}"
        )
    elif args.device == "gpu":
        os.system(
            f"sbatch --mail-user={args.mail} ./hydra/cluster_job_hydra_gpu_run_prebuild.sh {project_dir} {data_dir} {artifact_dir} {args.config} {args.mode}"
        )
    elif args.device == "cpu":
        os.system(
            f"sbatch --mail-user={args.mail} ./hydra/cluster_job_hydra_cpu_run_prebuild.sh {project_dir} {data_dir} {artifact_dir} {args.config} {args.mode}"
        )
    else:
        print("Device not recognized. Please choose gpu or cpu.")
        sys.exit(1)


if __name__ == '__main__':
    main()
