import ast
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
import pickle

from utils import dump_as_pickle


def prepare_difference_data(
    df: pd.DataFrame,
    idxs: dict,
    output_dir: str = None,
):
    pred_diffs = []
    attribution_diffs = []
    attribution_diffs_gt = []
    attribution_diffs_not_gt = []

    group_columns = [
        'model_name',
        'model_version',
        'model_repetition_number',
        'dataset_type',
        'attribution_method',
        'sentence_idx',
    ]
    for keys, group in tqdm(df.groupby(group_columns)):
        assert len(group) == 2, f"Expected two rows, got {len(group)}"
        info = {key: value for key, value in zip(group_columns, keys)}

        female = group[group["target"] == 0].iloc[0]
        male = group[group['target'] == 1].iloc[0]

        pred_diff = (
            female["pred_probabilities"][idxs["female"]]
            - male["pred_probabilities"][idxs["male"]]
        )
        pred_diffs.append(
            {
                **info,
                "pred_diff": pred_diff,
                "pred_female": female["pred_probabilities"][idxs["female"]],
                "pred_male": male["pred_probabilities"][idxs["male"]],
            }
        )

        for (
            female_word,
            male_word,
            female_attribution,
            male_attribution,
            gt,
        ) in zip(
            ast.literal_eval(female["sentence"]),
            ast.literal_eval(male["sentence"]),
            female["attribution"],
            male["attribution"],
            female["ground_truth"],
        ):
            attribution_diff = female_attribution - male_attribution

            diff_obj = {
                **info,
                "female_word": female_word.lower(),
                "male_word": male_word.lower(),
                "attribution_diff": attribution_diff,
                "attribution_female": female_attribution,
                "attribution_male": male_attribution,
            }

            attribution_diffs.append(diff_obj)

            if gt:
                attribution_diffs_gt.append(diff_obj)
            else:
                attribution_diffs_not_gt.append(diff_obj)

    pred_diffs_df = pd.DataFrame(pred_diffs)
    attribution_diffs_df = pd.DataFrame(attribution_diffs)
    attribution_diffs_gt_df = pd.DataFrame(attribution_diffs_gt)
    attribution_diffs_not_gt_df = pd.DataFrame(attribution_diffs_not_gt)

    dump_as_pickle(
        {
            "pred_diffs_df": pred_diffs_df,
            "attribution_diffs_df": attribution_diffs_df,
            "attribution_diffs_gt_df": attribution_diffs_gt_df,
            "attribution_diffs_not_gt_df": attribution_diffs_not_gt_df,
        },
        output_dir,
        "gender_differences.pkl",
    )


def load_differences(path: str, MODEL_NAME_MAP: dict[str, str]):
    with open(path, "rb") as f:
        data = pickle.load(f)

        pred_diffs_df = data["pred_diffs_df"]
        attribution_diffs_df = data["attribution_diffs_df"]
        attribution_diffs_gt_df = data["attribution_diffs_gt_df"]
        attribution_diffs_not_gt_df = data["attribution_diffs_not_gt_df"]

    pred_diffs_df["model_name"] = pred_diffs_df["model_name"].apply(
        lambda x: MODEL_NAME_MAP[x]
    )
    attribution_diffs_df["model_name"] = attribution_diffs_df["model_name"].apply(
        lambda x: MODEL_NAME_MAP[x]
    )
    attribution_diffs_gt_df["model_name"] = attribution_diffs_gt_df["model_name"].apply(
        lambda x: MODEL_NAME_MAP[x]
    )
    attribution_diffs_not_gt_df["model_name"] = attribution_diffs_not_gt_df[
        "model_name"
    ].apply(lambda x: MODEL_NAME_MAP[x])

    return (
        pred_diffs_df,
        attribution_diffs_df,
        attribution_diffs_gt_df,
        attribution_diffs_not_gt_df,
    )


def apply_prediction_test(pred_diffs_df: pd.DataFrame, test: str = "ttest"):
    group_by = [
        'model_name',
        'model_version',
        'dataset_type',
        'attribution_method',
        'model_repetition_number',
    ]

    # Predictions are model based and therefore same for all attribution methods
    # so we can just use the first one
    attributions_methods = pred_diffs_df["attribution_method"].unique()
    pred_diffs_df = pred_diffs_df[
        pred_diffs_df["attribution_method"] == attributions_methods[0]
    ]

    results = []
    for keys, group in pred_diffs_df.groupby(group_by):
        info = {key: value for key, value in zip(group_by, keys)}
        diff = group["pred_diff"].values

        alpha = 0.05
        if test == "ttest":
            mu = 0
            t_stat, p_value = stats.ttest_1samp(diff, mu)

            results.append(
                {
                    **info,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "reject": p_value < alpha,
                }
            )
        elif test == "wilcoxon":
            w_stat, p_value = stats.wilcoxon(diff)

            results.append(
                {
                    **info,
                    "w_stat": w_stat,
                    "p_value": p_value,
                    "reject": p_value < alpha,
                }
            )

    df = pd.DataFrame(results)

    return df


def apply_attribution_test(
    cur_df: pd.DataFrame, test: str, include_repetitions: bool = False
):
    results = []

    group_by = [
        'model_name',
        'model_version',
        'dataset_type',
        'attribution_method',
    ]

    if include_repetitions:
        group_by += ["model_repetition_number"]

    for keys, group in cur_df.groupby(group_by):
        info = {key: value for key, value in zip(group_by, keys)}
        diff = group["attribution_diff"].values

        alpha = 0.05

        if test == "ttest":
            mu = 0
            t_stat, p_value = stats.ttest_1samp(diff, mu)

            results.append(
                {
                    **info,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "reject": p_value < alpha,
                }
            )
        elif test == "wilcoxon":
            w_stat, p_value = stats.wilcoxon(diff)
            results.append(
                {
                    **info,
                    "w_stat": w_stat,
                    "p_value": p_value,
                    "reject": p_value < alpha,
                }
            )

    results_df = pd.DataFrame(results)

    return results_df
