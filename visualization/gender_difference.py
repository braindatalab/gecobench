import pandas as pd
from scipy import stats
import os
from scipy import stats
import seaborn as sns
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import matplotlib
from utils import list_intersection

from visualization.common import (
    MODEL_NAME_MAP,
    DATASET_NAME_MAP,
    METHOD_NAME_MAP,
    MODEL_ORDER,
    HUE_ORDER,
    ROW_ORDER,
)

from utils import load_pickle, generate_artifacts_dir, generate_evaluation_dir


def load_differences(path: str, MODEL_NAME_MAP: dict[str, str]):
    data = load_pickle(path)

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


def get_cutoff_cmap(alpha: float = 0.05, cmap_name: str = "magma"):
    vird = matplotlib.colormaps[cmap_name]
    new_colors = vird(
        np.concatenate(
            [
                np.linspace(0, 0.1, int(np.ceil(alpha * (256)))),
                np.linspace(0.6, 0.7, int(np.ceil((1 - alpha) * 256))),
            ]
        )
    )
    return ListedColormap(new_colors, name='cutoff')


def plot_prediction_heatmap(output_dir: str, df: pd.DataFrame, test: str = "ttest"):
    cmap = get_cutoff_cmap()
    gender_all = df[
        (df["model_version"] == "best") & (df["dataset_type"] == "gender_all")
    ].pivot(index="model_name", columns="model_repetition_number", values="p_value")

    gender_subj = df[
        (df["model_version"] == "best") & (df["dataset_type"] == "gender_subj")
    ].pivot(index="model_name", columns="model_repetition_number", values="p_value")

    max_value = max(gender_all.values.max(), gender_subj.values.max())

    fig, axs = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True, gridspec_kw={'width_ratios': [1, 1.2]}
    )
    sns.heatmap(
        gender_all,
        annot=True,
        fmt=".3f",
        ax=axs[0],
        vmax=max_value,
        cbar=False,
        cmap=cmap,
    )
    axs[0].set_title("$Dataset: D_A$")
    axs[0].set_ylabel("")
    axs[0].set_xlabel("")

    sns.heatmap(
        gender_subj,
        annot=True,
        fmt=".3f",
        ax=axs[1],
        vmax=max_value,
        cbar=True,
        cmap=cmap,
    )
    axs[1].set_title("$Dataset: D_S$")
    axs[1].set_ylabel("")
    axs[1].set_xlabel("")

    save_path = os.path.join(output_dir, "hypo_tests", test, f"prediction_diff.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    plt.close()


def plot_attribution_heatmap_row(
    df: pd.DataFrame, axs: list[plt.Axes], max_value: float = 1
) -> None:
    cmap = get_cutoff_cmap()

    df["attribution_method"] = df["attribution_method"].apply(
        lambda x: METHOD_NAME_MAP.get(x, x)
    )

    gender_all = df[df["dataset_type"] == "gender_all"].pivot(
        index="model_name", columns="attribution_method", values="p_value"
    )

    # Sort index by model_order
    gender_all.sort_index(
        key=lambda x: [MODEL_ORDER.index(model) for model in x], inplace=True
    )

    # Sort columns by HUE_ORDER
    gender_all = gender_all.reindex(columns=HUE_ORDER)

    gender_subj = df[df["dataset_type"] == "gender_subj"].pivot(
        index="model_name", columns="attribution_method", values="p_value"
    )

    # Sort index by model_order
    gender_subj.sort_index(
        key=lambda x: [MODEL_ORDER.index(model) for model in x], inplace=True
    )

    # Sort columns by HUE_ORDER
    gender_subj = gender_subj.reindex(columns=HUE_ORDER)

    sns.heatmap(
        gender_all,
        annot=True,
        fmt=".2f",
        ax=axs[0],
        vmax=max_value,
        cbar=False,
        cmap=cmap,
    )
    axs[0].set_title("Dataset: $D_A$")
    axs[0].set_ylabel("")
    axs[0].set_xlabel("")

    sns.heatmap(
        gender_subj,
        annot=True,
        fmt=".2f",
        ax=axs[1],
        vmax=max_value,
        cbar=True,
        cmap=cmap,
    )
    axs[1].set_title("Dataset: $D_S$")
    axs[1].set_ylabel("")
    axs[1].set_xlabel("")


def plot_attribution_heatmap(
    output_dir: str,
    results_df: pd.DataFrame,
    test: str,
    key: str = None,
) -> None:
    max_value = results_df["p_value"].values.max()

    fig, axs = plt.subplots(
        1, 2, figsize=(11, 6), sharey=True, gridspec_kw={'width_ratios': [1, 1.2]}
    )
    plot_attribution_heatmap_row(results_df, axs, max_value=max_value)

    fig.tight_layout()

    save_path = os.path.join(output_dir, "hypo_tests", test, f"attribution_{key}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_attribution_heatmap_with_rep(
    output_dir: str,
    results_df: pd.DataFrame,
    test: str,
    key: str = None,
) -> None:

    max_value = results_df["p_value"].values.max()

    model_repetitions = results_df["model_repetition_number"].unique()
    # Sort model repetitions
    model_repetitions = sorted(model_repetitions)

    fig, axs = plt.subplots(
        len(model_repetitions),
        2,
        figsize=(11, 6 * len(model_repetitions)),
        sharey=True,
        gridspec_kw={'width_ratios': [1, 1.2]},
    )

    for i, model_repetition in enumerate(model_repetitions):
        cur_results_df = results_df[
            results_df["model_repetition_number"] == model_repetition
        ]
        plot_attribution_heatmap_row(cur_results_df, axs[i], max_value=max_value)
        axs[i][0].set_title(f"Repetition {model_repetition}")

    fig.tight_layout()

    save_path = os.path.join(
        output_dir, "hypo_tests", test, f"attribution_rep_{key}.png"
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def generate_stats_table_attribution_with_rep(
    output_dir: str, attribution_diffs_df: pd.DataFrame, key: str
):
    results = []
    for (dataset, model, method), group in attribution_diffs_df.groupby(
        ["dataset_type", "model_name", "attribution_method"]
    ):
        for rep, group_rep in group.groupby("model_repetition_number"):
            abs_diffs = np.abs(group_rep["attribution_diff"])

            results.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "rep": rep,
                    "method": method,
                    "mean": abs_diffs.mean(),
                    "std": abs_diffs.std(),
                }
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(output_dir, f"attribution_diffs_{key}.csv"),
        index=False,
    )

    return results_df


def generate_stats_table_pred(output_dir: str, pred_diffs_df: pd.DataFrame):
    results = []
    for dataset, group_ds in pred_diffs_df.groupby("dataset_type"):
        for model, group in group_ds.groupby("model_name"):

            stds = []
            means = []
            for _, group_rep in group.groupby("model_repetition_number"):
                abs_diffs = np.abs(group_rep["pred_diff"])
                means += [abs_diffs.mean()]
                stds += [abs_diffs.std()]

            results.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "mean": np.mean(means),
                    "mean_std": np.std(means),
                    "std": np.mean(stds),
                    "std_std": np.std(stds),
                }
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        os.path.join(output_dir, "prediction_diff.csv"),
        index=False,
    )


def plot_word_attribution_difference_point(output_dir: str, df: pd.DataFrame, key: str):
    height = 2.5

    try:
        df["dataset"] = df["dataset"].map(lambda x: DATASET_NAME_MAP.get(x, x))
    except KeyError as e:
        print(df.columns)
        raise e

    def _plot_postprocessing(g):
        g.set_titles(row_template="Dataset: {row_name}", size=12)
        for k in range(g.axes.shape[0]):
            for j in range(g.axes.shape[1]):
                g.axes[k, j].grid(alpha=0.8, linewidth=0.5)
                g.axes[k, j].set_xlabel('', fontsize=12)
                g.axes[k, j].set_ylabel(g.axes[k, j].get_ylabel(), fontsize=12)

                for label in (
                    g.axes[k, j].get_xticklabels() + g.axes[k, j].get_yticklabels()
                ):
                    label.set_fontsize(12)

        # Set legend font size
        g._legend.set_title('XAI Method', prop={'size': 12})
        for t in g._legend.texts:
            t.set_fontsize(12)

    g = sns.catplot(
        data=df,
        x="Model",
        # y="Attribution Difference per word",
        y="Average Attribution Difference",
        hue="XAI Method",
        row="dataset",
        kind="point",
        hue_order=list_intersection(HUE_ORDER, df["XAI Method"].unique()),
        order=MODEL_ORDER,
        row_order=ROW_ORDER,
        palette=sns.color_palette(palette='pastel'),
        height=height,
        aspect=9.5 / height,
        dodge=0.5,
        # margin_titles=True,
        legend_out=True,
        linestyle='',
        err_kws={
            "linewidth": 1.5,
            "color": "black",
            "zorder": 0,
        },
        markeredgecolor='grey',
        markeredgewidth=0.5,
    )

    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.43, -0.13),
        ncol=5,
        frameon=True,
    )

    _plot_postprocessing(g=g)

    save_path = os.path.join(output_dir, f"word_attribution_diff_{key}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def create_gender_difference_plots(base_output_dir: str, config: dict):
    logger.info("Generating gender difference plots")

    results_path = os.path.join(
        generate_artifacts_dir(config),
        generate_evaluation_dir(config),
        "gender_differences.pkl",
    )

    (
        pred_diffs_df,
        attribution_diffs_df,
        attribution_diffs_gt_df,
        attribution_diffs_not_gt_df,
    ) = load_differences(results_path, MODEL_NAME_MAP=MODEL_NAME_MAP)

    # Exclude methods not defined in config
    for method in attribution_diffs_df["attribution_method"].unique():
        if method not in config["xai"]["methods"]:
            attribution_diffs_df = attribution_diffs_df[
                attribution_diffs_df["attribution_method"] != method
            ]
            attribution_diffs_gt_df = attribution_diffs_gt_df[
                attribution_diffs_gt_df["attribution_method"] != method
            ]
            attribution_diffs_not_gt_df = attribution_diffs_not_gt_df[
                attribution_diffs_not_gt_df["attribution_method"] != method
            ]

    # Generate difference tables & plots
    generate_stats_table_pred(base_output_dir, pred_diffs_df)

    for key, df in [
        ("all", attribution_diffs_df),
        ("gt", attribution_diffs_gt_df),
        ("not_gt", attribution_diffs_not_gt_df),
    ]:
        df_with_rep = generate_stats_table_attribution_with_rep(
            base_output_dir, df, key
        )

        df_with_rep["method"] = df_with_rep["method"].apply(
            lambda x: METHOD_NAME_MAP.get(x, x)
        )
        df_with_rep.rename(
            columns={
                "method": "XAI Method",
                "model": "Model",
                "mean": "Average Attribution Difference",
            },
            inplace=True,
        )

        plot_word_attribution_difference_point(base_output_dir, df_with_rep, key)

    # Generate hypothesis test plots
    for test in ["ttest", "wilcoxon"]:
        if len(pred_diffs_df) > 0:
            pred_results = apply_prediction_test(pred_diffs_df, test)
            plot_prediction_heatmap(base_output_dir, pred_results, test)

        results_all = apply_attribution_test(attribution_diffs_df, test)
        results_gt = apply_attribution_test(attribution_diffs_gt_df, test)
        results_not_gt = apply_attribution_test(attribution_diffs_not_gt_df, test)

        for results, key in [
            (results_all, "all"),
            (results_gt, "gt"),
            (results_not_gt, "not_gt"),
        ]:
            plot_attribution_heatmap(base_output_dir, results, test, key)

        results_all_rep = apply_attribution_test(
            attribution_diffs_df, test, include_repetitions=True
        )
        results_gt_rep = apply_attribution_test(
            attribution_diffs_gt_df, test, include_repetitions=True
        )
        results_not_gt_rep = apply_attribution_test(
            attribution_diffs_not_gt_df, test, include_repetitions=True
        )

        for results, key in [
            (results_all_rep, "all"),
            (results_gt_rep, "gt"),
            (results_not_gt_rep, "not_gt"),
        ]:
            plot_attribution_heatmap_with_rep(base_output_dir, results, test, key)

    logger.info("Finished generating gender difference plots")
