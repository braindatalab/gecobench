import argparse

from data.main import main as data_main
from utils import load_json_file


def get_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        dest='config',
        required=True,
        help='File path to data config.',
        type=str,
        default=1,
    )
    return parser.parse_args()


def main(config_path: str):
    project_config = load_json_file(file_path=config_path)
    data_main(config=project_config)


if __name__ == '__main__':
    args = get_command_line_arguments()
    main(config_path=args.config)
