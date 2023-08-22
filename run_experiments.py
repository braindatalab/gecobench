import argparse

from data.main import main as data_main
from training.main import main as training_main
from xai.main import main as xai_main
from evaluation.main import main as evaluation_main
from utils import load_json_file


def get_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config",
        required=True,
        help="File path to project config.",
        type=str,
        default=1,
    )
    parser.add_argument(
        "--mode",
        dest="mode",
        required=True,
        help="Modes: filtering, training, xai, evaluation, visualization.",
        type=str,
        default=1,
    )
    return parser.parse_args()


def main(config_path: str, mode: str):
    project_config = load_json_file(file_path=config_path)
    Modes[mode](config=project_config)


Modes = {
    "data": data_main,
    "training": training_main,
    "xai": xai_main,
    "evaluation": evaluation_main,
}

if __name__ == "__main__":
    args = get_command_line_arguments()
    main(config_path=args.config, mode=args.mode)
