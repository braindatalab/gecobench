import argparse
import os
import subprocess
import tarfile
from os.path import basename, join
from time import sleep
from paramiko.client import SSHClient
from datetime import datetime

from dotenv import load_dotenv


def today_formatted() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def append_date(s: str) -> str:
    date = today_formatted()
    return f'{s}-{date}'


load_dotenv()

hydra_base_dir = (
    os.path.dirname(os.environ.get('HYDRA_PROJECT_DIR')) or '/home/space/uniml/rick'
)
hpc_base_dir = os.environ.get('HPC_BASE_DIR') or '/home/users/r/rick'
known_hosts_path = os.environ.get('KNOWN_HOSTS') or '/home/rick/.ssh/known_hosts'
user_name = os.environ.get('HYDRA_SSH_USER') or 'rick'

print("User name: ", user_name)

CLUSTER_NAMES = ['hydra', 'hpc']
CLUSTER_DOMAINS = dict(hydra='hydra.ml.tu-berlin.de', hpc='gateway.hpc.tu-berlin.de')
CLUSTER_BASE_DIRS = dict(hydra=hydra_base_dir, hpc=hpc_base_dir)

ignore_list = [".pkl", "artifacts", ".git", ".venv", ".env", "__pycache__", "wandb"]


def create_tarfile(source_dir: str, output_filename: str) -> None:
    def filter_function(tarinfo):
        print(tarinfo.name)
        if any([ignore in tarinfo.name for ignore in ignore_list]):
            return None
        else:
            return tarinfo

    with tarfile.open(name=output_filename, mode='w:gz') as tar:
        tar.add(name=source_dir, arcname=basename(source_dir), filter=filter_function)


def copy_file_to_cluster(cluster_name: str, source_path: str, target_path: str) -> None:
    command = ['scp', source_path, f'{user_name}@{cluster_name}:{target_path}']
    _ = subprocess.run(command, stdout=subprocess.PIPE)


def remove_local_file(file_path: str) -> None:
    command = ['rm', file_path]
    _ = subprocess.run(command, stdout=subprocess.PIPE)


def pause_script(seconds: int):
    print(f'Wait {seconds} seconds.')
    sleep(seconds)


def get_ssh_connection_to_cluster(cluster_name: str) -> SSHClient:
    client = SSHClient()
    client.load_host_keys(filename=known_hosts_path)
    client.connect(CLUSTER_DOMAINS[cluster_name], username=user_name)
    return client


def extract_tar_file(ssh_client: SSHClient, file_path: str) -> str:
    file_path_split = file_path.split('/')
    target_dir = '/'.join(file_path_split[:-1])
    target_filename = file_path_split[-1].split('.')[0]
    target_path = f'{target_dir}/{target_filename}'
    command_mkdir = f'mkdir {target_path}'
    command_untar = f'tar -xf {file_path} -C {target_path} --strip-components 1'
    ssh_client.exec_command(command=f'{command_mkdir} && {command_untar}')
    return join(target_dir, target_filename)


def main(cluster_name: str) -> None:
    current_dir = os.getcwd()
    filename_compressed_code = f'{append_date(s=basename(current_dir))}.tar.gz'
    print('Create tar archive of source code.')
    create_tarfile(source_dir=current_dir, output_filename=filename_compressed_code)

    print('Copy code to cluster.')
    target_path_compressed_code = join(
        CLUSTER_BASE_DIRS[cluster_name], filename_compressed_code
    )
    copy_file_to_cluster(
        cluster_name=cluster_name,
        source_path=filename_compressed_code,
        target_path=target_path_compressed_code,
    )

    print('Extract tar file on cluster.')
    pause_script(seconds=5)
    _ = extract_tar_file(
        ssh_client=get_ssh_connection_to_cluster(cluster_name=cluster_name),
        file_path=target_path_compressed_code,
    )

    pause_script(seconds=5)
    remove_local_file(file_path=filename_compressed_code)
    pause_script(seconds=5)


def get_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cluster',
        dest='cluster',
        required=True,
        help='Name of cluster',
        type=str,
        default=1,
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = get_command_line_arguments()
    main(cluster_name=args.cluster)
