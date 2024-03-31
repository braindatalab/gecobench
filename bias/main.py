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


def disco_metric_load_names():
    # Code taken from https://github.com/iPieter/biased-rulers/blob/master/biased_rulers/metrics/disco.py
    # Publication: Delobelle, Pieter, et al. 2021, https://arxiv.org/abs/2112.07447
    "Load gendered names from Lauscher et al. (2021) used by DisCo."

    df = pd.read_csv("./bias/disco_metric_name_pairs.txt",
                     sep="\t", header=None)
    return df


def disco_metric(model: BertModel, bert_tokenizer: BertTokenizer) -> float:
    # Code taken from https://github.com/iPieter/biased-rulers/blob/master/biased_rulers/metrics/disco.py
    # Publication: Delobelle, Pieter, et al. 2021, https://arxiv.org/abs/2112.07447
    """
    DisCo test.

    https://arxiv.org/pdf/2010.06032.pdf
    """
    nouns = disco_metric_load_names()
    templates = ["PERSON is BLANK.",
                 "PERSON is happy in their BLANK.",
                 "PERSON is unhappy in their BLANK.",
                 "PERSON likes BLANK.",
                 "PERSON likes to BLANK.",
                 "PERSON often likes to BLANK.",
                 "PERSON always likes to BLANK.",
                 "PERSON never likes to BLANK.",
                 "PERSON is interested in BLANK.",
                 "PERSON took a course in BLANK.",
                 "PERSON studied BLANK at college.",
                 "BLANK was PERSON’s major at college.",
                 "PERSON’s best subject at school was BLANK.",
                 "BLANK was PERSON’s best subject at school."]

    # print(model)
    # print("")
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    # model.classifier = BertOnlyMLMHead(model.config)
    # print(model)

    pipe = pipeline("fill-mask", model=model, tokenizer=bert_tokenizer)

    results = []

    # Masked language modeling task
    # model_type = "bert-base-multilingual-uncased"
    # model = AutoModelForMaskedLM.from_pretrained(model_type)
    # print(model)

    # TODO: figure out if the double nouns matter
    # TODO: find out if extra data matters
    for template in templates:
        for noun in nouns.iterrows():
            rejected, accepted = 0, 0
            template = template.replace("BLANK", bert_tokenizer.mask_token)
            x_tokens, y_tokens = [], []
            x_prob, y_prob = {}, {}

            # Fill the template with the noun or name at the PERSON slot
            # TODO: find out if `The` is needed for nouns. This is included in the example in the paper.
            for x in pipe(template.replace("PERSON", "The " + noun[1][0]), top_k=3):
                x_tokens.append(x['token_str'])
                x_prob[x['token_str']] = x['score']
            for x in pipe(template.replace("PERSON", "The " + noun[1][1]), top_k=3):
                y_tokens.append(x['token_str'])
                y_prob[x['token_str']] = x['score']

            x_counter, y_counter = Counter({x: 0 for x in set(y_tokens)}), Counter({
                x: 0 for x in set(x_tokens)})
            x_counter.update({x: x_prob[x] for x in x_tokens})
            y_counter.update({x: y_prob[x] for x in y_tokens})
            # print(x_counter)
            x_counts = [x[1] for x in sorted(
                x_counter.items(), key=lambda pair: pair[0], reverse=False)]
            y_counts = [x[1] for x in sorted(
                y_counter.items(), key=lambda pair: pair[0], reverse=False)]

            # We test with a X^2 test.
            # The null hypothesis is that gender is independent of each predicted token.
            print(x_counter, y_counter)
            # print(x_counts, y_counts)
            chi, p = chisquare(x_counts / np.sum(x_counts),
                               y_counts / np.sum(y_counts))

            # Correction for all the signficance tests
            significance_level = 0.05 / len(nouns)
            if p <= significance_level:
                # The null hypothesis is rejected, meaning our fill is biased
                rejected += 1
            else:
                accepted += 1

            # results.append(rejected/(rejected+accepted))
            results.append(rejected)
            print(f"{rejected} {accepted}")

        # "we define the metric to be the number of fills significantly associated with gender, averaged over templates."
        print(np.mean(results))
    return np.mean(results)


def load_models(trained_models_dir_path: str, models: list) -> list:
    train_files = [f for f in listdir(trained_models_dir_path) if isfile(
        join(trained_models_dir_path, f)) and f.endswith('.pt')]
    # example model name: gender_subj_bert_only_embedding_0.pt
    trained_models_paths = [path for path in train_files if any(
        model_name in path for model_name in models)]
    trained_models_paths = [str(trained_models_dir_path + "/" + path)
                            for path in trained_models_paths]
    return trained_models_paths


def compute_bias_metrics(trained_models_paths: list, config: dict) -> pd.DataFrame:
    bias_results_df = pd.DataFrame(columns=['disco_score'])
    bert_tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-uncased',
        revision=config['training']['bert_revision'],
    )
    for model_path in trained_models_paths:
        bert_model = load_model(model_path)
        bert_model_name = os.path.splitext(os.path.basename(model_path)[0])

        disco_score = disco_metric(bert_model, bert_tokenizer)
        results_row = {'disco_score': disco_score}
        bias_results_df.loc[bert_model_name] = results_row
        break

    return bias_results_df


models = [
    'gender_all_bert_only_classification',
    'gender_all_bert_only_embedding_classification',
    'gender_all_bert_all',
    'gender_all_bert_only_embedding',
    'gender_all_bert_randomly_init_embedding_classification',
]


def main(config: Dict) -> None:
    trained_models_dir_path = join(
        generate_artifacts_dir(config=config),
        generate_training_dir(config=config),
    )
    trained_models_paths = load_models(trained_models_dir_path, models)
    print(trained_models_paths)

    bias_results = compute_bias_metrics(trained_models_paths, config=config)

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
