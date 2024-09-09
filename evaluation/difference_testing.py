import ast
import pandas as pd
from tqdm import tqdm
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
            female["pred_probs"][idxs["female"]]
            - male["pred_probs"][idxs["male"]]
        )
        pred_diffs.append(
            {
                **info,
                "pred_diff": pred_diff,
                "pred_female": female["pred_probs"][idxs["female"]],
                "pred_male": male["pred_probs"][idxs["male"]],
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
