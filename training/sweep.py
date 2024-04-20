import wandb

from training.main import load_dataset, TrainModel
from utils import load_json_file, generate_data_dir


def main(config: dict) -> None:
    wandb.login()

    # Only one model repetition is supported for sweeps
    config["training"]["num_training_repetitions"] = 1

    sweep_config = config["sweep"]
    model_name = sweep_config["config"]["parameters"]["model_name"]["value"]
    dataset_name = sweep_config["config"]["parameters"]["dataset_name"]["value"]
    num_labels = 2

    dataset = load_dataset(config, dataset_name)
    params = config['training']['models'][model_name]

    def run_sweep():
        wandb.init()
        model_config = wandb.config
        merged_params = {**params, **model_config}
        TrainModel[model_name](dataset, dataset_name, num_labels, merged_params, config)

    if "id" in sweep_config:
        # We already have a sweep id, so we can just run the sweep
        wandb.agent(
            sweep_config["id"],
            project=config["general"]["wandb"]["project"],
            entity=config["general"]["wandb"]["entity"],
            function=run_sweep,
        )

    else:
        # We need to create a new sweep
        sweep_id = wandb.sweep(
            sweep_config["config"],
            project=config["general"]["wandb"]["project"],
            entity=config["general"]["wandb"]["entity"],
        )
        wandb.agent(sweep_id, function=run_sweep)
