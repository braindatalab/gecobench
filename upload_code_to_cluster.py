import argparse
import json
import os
import subprocess
import tarfile
from os.path import basename, join
from time import sleep
from paramiko.client import SSHClient

from utils import load_json_file, append_date


def create_tarfile(source_dir: str, output_filename: str) -> None:
    def filter_function(tarinfo):
        if '.pkl' in tarinfo.name or 'artifacts' in tarinfo.name:
            return None
        else:
            return tarinfo

    with tarfile.open(name=output_filename, mode='w:gz') as tar:
        tar.add(name=source_dir, arcname=basename(source_dir), filter=filter_function)


def copy_file_to_cluster(source_path: str, target_path: str, user_name: str = 'rick') -> None:
    command = ['scp', source_path, f'{user_name}@hydra:{target_path}']
    _ = subprocess.run(command, stdout=subprocess.PIPE)


def remove_local_file(file_path: str) -> None:
    command = ['rm', file_path]
    _ = subprocess.run(command, stdout=subprocess.PIPE)


def pause_script(seconds: int):
    print(f'Wait {seconds} seconds.')
    sleep(seconds)


def get_ssh_connection_to_cluster() -> SSHClient:
    client = SSHClient()
    client.load_host_keys(filename='/home/rick/.ssh/known_hosts')
    client.connect('hydra.ml.tu-berlin.de')
    return client


def extract_tar_file(ssh_client: SSHClient, file_path: str) -> str:
    file_path_split = file_path.split('/')
    target_dir = '/'.join(file_path_split[:-1])
    target_filename = file_path_split[-1].split('.')[0]
    command = f'tar -xf {file_path} -C {target_dir} --one-top-level={target_filename} --strip-components 1'
    ssh_client.exec_command(command=command)
    return join(target_dir, target_filename)


def main() -> None:
    config_cluster = load_json_file(file_path='cluster_config.json')
    current_dir = os.getcwd()

    filename_compressed_code = f'{append_date(s=basename(current_dir))}.tar.gz'
    print('Create tar archive of source code.')
    create_tarfile(source_dir=current_dir, output_filename=filename_compressed_code)

    print('Copy code to cluster.')
    target_path_compressed_code = join(config_cluster['base_dir'], filename_compressed_code)
    copy_file_to_cluster(source_path=filename_compressed_code, target_path=target_path_compressed_code)

    print('Extract tar file on cluster.')
    pause_script(seconds=5)
    client = get_ssh_connection_to_cluster()
    target_dir = extract_tar_file(ssh_client=client, file_path=target_path_compressed_code)

    pause_script(seconds=5)
    remove_local_file(file_path=filename_compressed_code)
    pause_script(seconds=5)


if __name__ == '__main__':
    main()
