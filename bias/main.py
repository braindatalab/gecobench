import numpy as np
import pandas as pd
import torch

import os
from os import listdir
from os.path import join, isfile, join
from typing import Dict

from transformers import BertTokenizer, BertModel, pipeline, AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertPooler, BertOnlyMLMHead
from collections import Counter
from scipy.stats import chi2_contingency, chisquare

from utils import (
    filter_xai_datasets,
    generate_training_dir,
    load_json_file,
    load_jsonl_as_df,
    load_pickle,
    dump_as_pickle,
    generate_xai_dir,
    append_date,
    load_model,
    load_jsonl_as_dict,
    generate_data_dir,
    generate_artifacts_dir,
)


# TODO: Implement occurence matrix S 
# Load datasets
# Identify gender words (female: she, her, women, etc.)
# Iterate through all sentences
    #  Iteratre through 


def main(config: Dict) -> None:
    trained_models_dir_path = join(
        generate_artifacts_dir(config=config),
        generate_training_dir(config=config),
    )


    # logger.info(f"Calculate evaluation scores.")
    # evaluation_results = evaluate(data=evaluation_data)
    # filename = config["evaluation"]["evaluation_records"]
    # logger.info(f"Output path: {join(artifacts_dir, evaluation_output_dir, filename)}")
    # dump_as_pickle(
    #     data=evaluation_results,
    #     output_dir=join(artifacts_dir, evaluation_output_dir),
    #     filename=filename,
    # )


if __name__ == '__main__':
    main(config={})
