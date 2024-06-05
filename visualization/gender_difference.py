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
    color_range = 2028
    first_section = int(np.floor((alpha) * color_range))
    second_section = color_range - first_section
    new_colors = vird(
        np.concatenate(
            [
                np.linspace(0, 0.1, first_section),
                np.linspace(0.6, 0.7, second_section),
            ]
        )
    )
    return ListedColormap(new_colors, name='cutoff')


def plot_prediction_heatmap(output_dir: str, df: pd.DataFrame, test: str = "ttest"):
    gender_all = df[
        (df["model_version"] == "best") & (df["dataset_type"] == "gender_all")
    ].pivot(index="model_name", columns="model_repetition_number", values="p_value")

    gender_all.sort_index(
        key=lambda x: [MODEL_ORDER.index(model) for model in x], inplace=True
    )

    gender_subj = df[
        (df["model_version"] == "best") & (df["dataset_type"] == "gender_subj")
    ].pivot(index="model_name", columns="model_repetition_number", values="p_value")

    gender_subj.sort_index(
        key=lambda x: [MODEL_ORDER.index(model) for model in x], inplace=True
    )

    cmap = get_cutoff_cmap()

    fig, axs = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True, gridspec_kw={'width_ratios': [1, 1.2]}
    )
    sns.heatmap(
        gender_all,
        annot=True,
        fmt=".3f",
        ax=axs[0],
        vmin=0,
        vmax=1.0,
        cbar=False,
        cmap=cmap,
    )
    axs[0].set_title("$Dataset: D_A$")
    axs[0].set_ylabel("")
    axs[0].set_xlabel("")

    g = sns.heatmap(
        gender_subj,
        annot=True,
        fmt=".3f",
        ax=axs[1],
        vmin=0,
        vmax=1.0,
        cbar=True,
        cmap=cmap,
    )
    axs[1].set_title("$Dataset: D_S$")
    axs[1].set_ylabel("")
    axs[1].set_xlabel("")
    g.collections[0].colorbar.set_label("p-value")
    g.collections[0].colorbar.set_ticks([0, 0.05, 1])

    save_path = os.path.join(output_dir, "hypo_tests", test, f"prediction_diff.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path)
    plt.close()


def plot_attribution_heatmap_row(
    df: pd.DataFrame, axs: list[plt.Axes], max_value: float = 1, y_label: str = ""
) -> None:

    def truncate_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
        return df.applymap(
            lambda x: (
                np.trunc(x * 10**decimals) / 10 ** decimals if not pd.isnull(x) else x
            )
        )

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
    gender_all = truncate_df(gender_all)

    gender_subj = df[df["dataset_type"] == "gender_subj"].pivot(
        index="model_name", columns="attribution_method", values="p_value"
    )

    # Sort index by model_order
    gender_subj.sort_index(
        key=lambda x: [MODEL_ORDER.index(model) for model in x], inplace=True
    )

    # Sort columns by HUE_ORDER
    gender_subj = gender_subj.reindex(columns=HUE_ORDER)
    gender_subj = truncate_df(gender_subj)

    cmap = get_cutoff_cmap()

    sns.heatmap(
        gender_all,
        annot=True,
        fmt=".2f",
        ax=axs[0],
        vmin=0,
        vmax=1,
        cbar=False,
        cmap=cmap,
    )
    axs[0].set_title("Dataset: $D_A$", fontsize=16)
    axs[0].set_ylabel(y_label, fontsize=14)
    axs[0].set_xlabel("")

    g = sns.heatmap(
        gender_subj,
        annot=True,
        fmt=".2f",
        ax=axs[1],
        vmin=0,
        vmax=1,
        cbar=True,
        cmap=cmap,
    )
    axs[1].set_title("Dataset: $D_S$", fontsize=16)
    axs[1].set_ylabel("")
    axs[1].set_xlabel("")
    g.collections[0].colorbar.set_label("p-value", fontsize=16)
    g.collections[0].colorbar.set_ticks([0, 0.05, 1])
    g.collections[0].colorbar.ax.tick_params(labelsize=14)

    for ax in axs:
        # Set font size of tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(14)

        # Rotate y labels
        labels = [x.get_text() for x in ax.get_yticklabels()]
        if len(labels) > 0:
            print(labels)
            ax.set_yticklabels(labels, rotation=0, fontsize=14)


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
        plot_attribution_heatmap_row(
            cur_results_df,
            axs[i],
            max_value=max_value,
            y_label=f"Rep: {model_repetition}",
        )

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


def plot_prediction_difference(
    base_output_dir: str, pred_diffs_df: pd.DataFrame, **kwargs
):
    # We have one row for every attribution method, but the prediction difference is the same for all
    # so we can just use the first one.
    df = pred_diffs_df[
        (
            pred_diffs_df["attribution_method"]
            == pred_diffs_df["attribution_method"].unique()[0]
        )
        & (pred_diffs_df["model_version"] == "best")
    ]
    df["dataset_type"] = df["dataset_type"].map(lambda x: DATASET_NAME_MAP.get(x, x))

    df_pos = df[df["pred_diff"] > 0]
    df_neg = df[df["pred_diff"] <= 0]

    fig, ax = plt.subplots(figsize=(10, 5))

    def make_errorbar(plot_negative):
        def plot_errorbar(x):
            # Get the first index of the group to get the model and dataset
            idx = x.index[0]
            entry = df.loc[idx]

            cur_abs_df = df[
                (df["model_name"] == entry["model_name"])
                & (df["dataset_type"] == entry["dataset_type"])
            ]
            cur_abs_df = cur_abs_df[cur_abs_df["pred_diff"] != 0]

            cur_abs_diff = np.mean(np.abs(cur_abs_df["pred_diff"]))

            if plot_negative:
                return (0, cur_abs_diff)
            else:
                return (-cur_abs_diff, 0)

        return plot_errorbar

    for idx, (cur_df, colors) in enumerate(
        [
            (df_pos, sns.color_palette("pastel")),
            (df_neg, sns.color_palette()),
        ]
    ):

        x_label = "Model"
        y_label = "P(female) - P(male)"
        hue_label = "Dataset"
        cur_df = cur_df.rename(
            columns={
                "model_name": x_label,
                "pred_diff": y_label,
                "dataset_type": hue_label,
            }
        )

        g = sns.barplot(
            data=cur_df,
            x=x_label,
            y=y_label,
            hue=hue_label,
            ax=ax,
            palette=colors,
            legend=idx == 0,
            errorbar=make_errorbar(idx == 1),
        )
        g.set_xlabel("")
        g.set_ylabel(y_label, fontsize=12)
        for label in g.get_xticklabels() + g.get_yticklabels():
            label.set_fontsize(12)

        if idx == 0:
            leg = g.legend(title=hue_label)
            # Set title font size
            leg.set_title(title=hue_label, prop={'size': 12})
            for t in leg.texts:
                t.set_fontsize(12)

    plt.savefig(
        os.path.join(base_output_dir, "bias_prediction_diff.png"),
        bbox_inches='tight',
        dpi=300,
    )
    plt.close()


def plot_test_heatmaps(
    base_output_dir: str,
    pred_diffs_df: pd.DataFrame,
    attribution_diffs_df: pd.DataFrame,
    attribution_diffs_gt_df: pd.DataFrame,
    attribution_diffs_not_gt_df: pd.DataFrame,
):
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


def plot_means_stats(
    base_output_dir: str,
    pred_diffs_df: pd.DataFrame,
    attribution_diffs_df: pd.DataFrame,
    attribution_diffs_gt_df: pd.DataFrame,
    attribution_diffs_not_gt_df: pd.DataFrame,
):
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

    visualization_methods = dict(
        test_heatmaps=plot_test_heatmaps,
        stats=plot_means_stats,
        prediction_difference=plot_prediction_difference,
    )

    plot_types = config['visualization']['visualizations']['gender_difference']
    for plot_type in plot_types:
        logger.info(f'Type of plot: {plot_type}')
        v = visualization_methods.get(plot_type, None)
        if v is None:
            continue
        v(
            base_output_dir=base_output_dir,
            pred_diffs_df=pred_diffs_df,
            attribution_diffs_df=attribution_diffs_df,
            attribution_diffs_gt_df=attribution_diffs_gt_df,
            attribution_diffs_not_gt_df=attribution_diffs_not_gt_df,
        )

    logger.info("Finished generating gender difference plots")
