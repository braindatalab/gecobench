

import pandas as pd
from scipy import stats


def apply_wilcoxon_test(data: pd.DataFrame):
    group_by = [
        'Model',
        'model_version',
        'Dataset',
        'XAI Method',
        # 'model_repetition_number',
    ]

    results = []
    for keys, group in data.groupby(group_by):
        info = {key: value for key, value in zip(group_by, keys)}
        diff = group["median(RMA)"].values

        alpha = 0.05

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
