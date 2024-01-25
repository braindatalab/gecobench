from typing import Dict
import os
import pandas as pd
import numpy as np
from data.utils.kaggle import download_kaggle


def create_train_val_datasets(config: dict, split: float):
    df = pd.read_csv(os.path.join(self.data_dir, "movie.csv"))

    # Drop nan values
    df = df.dropna()

    # Shuffle
    idxs = np.arange(len(df))
    idxs = np.random.permutation(idxs)
    train_idxs = idxs[: int(len(idxs) * split)]
    val_idxs = idxs[int(len(idxs) * split) :]

    train_df = df.iloc[train_idxs]
    val_df = df.iloc[val_idxs]

    # Save to csv
    train_df.to_csv(self.train_df_path, index=False)
    val_df.to_csv(self.val_df_path, index=False)


def prepare_imdb_sentiment_data(config: Dict, data_output_dir: str):
    ds_config = config["data"]["datasets"]["sentiment_twitter"]

    download_data_dir = download_kaggle(
        data_output_dir=data_output_dir,
        dataset_subdir="sentiment_imdb",
        ds_config=ds_config["kaggle"],
    )
