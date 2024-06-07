from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from os.path import join, exists
from pathlib import Path
from typing import Tuple

from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from pandas.core.groupby.generic import DataFrameGroupBy
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from common import DatasetKeys, EvaluationResult, SaveVersion
from utils import list_intersection

from visualization.gender_difference import create_gender_difference_plots
from visualization.common import (
    MODEL_NAME_MAP,
    DATASET_NAME_MAP,
    METHOD_NAME_MAP,
    METRIC_NAME_MAP,
    MODEL_NAME_HTML_MAP,
    MODEL_ORDER,
    HUE_ORDER,
    ROW_ORDER,
    GENDER,
)

from utils import (
    generate_visualization_dir,
    generate_evaluation_dir,
    load_pickle,
    load_jsonl_as_df,
    generate_training_dir,
    generate_xai_dir,
    generate_project_dir,
    generate_data_dir,
    generate_artifacts_dir,
)


MOST_COMMON_XAI_ATTRIBUTION_PLOT_TYPES = dict(
    accumulated="most_common_xai_attributions",
    tf_idf="most_common_xai_attributions_tf_idf",
    freq="most_common_xai_attributions_freq",
)


def compute_average_score_per_repetition(data: pd.DataFrame) -> pd.DataFrame:
    results = list()
    for k, df_dataset_type in data.groupby(by='Dataset'):
        for l, df_model_type in df_dataset_type.groupby(by='Model'):
            for j, df_repetition in df_model_type.groupby(by='model_repetition_number'):
                for i, df_xai_method in df_repetition.groupby(by='XAI Method'):
                    average_scores = (
                        df_xai_method._get_numeric_data().aggregate(['mean']).iloc[0, :]
                    )
                    first_row = deepcopy(df_xai_method.iloc[0, :])
                    first_row.loc[average_scores.index.values] = np.nan
                    results += [first_row.combine_first(average_scores)]

    return pd.concat(results, axis=1).T


def plot_evaluation_results(
    data: pd.DataFrame,
    metric: str,
    model_version: str,
    result_type: str,
    base_output_dir: str,
) -> None:
    def _plot_postprocessing(g):
        g.set_titles(row_template="Dataset: {row_name}", size=12)
        for k in range(g.axes.shape[0]):
            for j in range(g.axes.shape[1]):
                g.axes[k, j].grid(alpha=0.8, linewidth=0.5)

                if 0 == k and 'top_k_precision' == metric:
                    g.axes[k, j].set_ylabel(
                        f'Average {METRIC_NAME_MAP[metric]}',
                        fontsize=12,
                    )
                else:
                    g.axes[k, j].set_ylabel(
                        f'{METRIC_NAME_MAP[metric]}',
                        fontsize=12,
                    )
                g.axes[k, j].set_xlabel('')
                g.axes[k, j].set_ylim(0, 1)
                g.axes[k, j].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])

                for label in (
                    g.axes[k, j].get_xticklabels() + g.axes[k, j].get_yticklabels()
                ):
                    label.set_fontsize(12)

        # Set legend font size
        g._legend.set_title('XAI Method', prop={'size': 12})
        for t in g._legend.texts:
            t.set_fontsize(12)

    data['mapped_model_name'] = data['model_name'].map(lambda x: MODEL_NAME_MAP[x])
    data['dataset_type'] = data['dataset_type'].map(lambda x: DATASET_NAME_MAP[x])
    data['attribution_method'] = data['attribution_method'].map(
        lambda x: METHOD_NAME_MAP.get(x, x)
    )

    data = data.rename(
        columns={
            "mapped_model_name": "Model",
            "dataset_type": "Dataset",
            "attribution_method": "XAI Method",
            **METRIC_NAME_MAP,
        }
    )

    if metric != 'top_k_precision':
        average_data = compute_average_score_per_repetition(data=data)
        datasets = [('', data), ('averaged', average_data)]

        height = 2.5
        for s, d in datasets:
            g = sns.catplot(
                data=d,
                x='Model',
                y=METRIC_NAME_MAP[metric],
                order=MODEL_ORDER,
                hue_order=list_intersection(HUE_ORDER, d['XAI Method'].unique()),
                row_order=ROW_ORDER,
                hue='XAI Method',
                row='Dataset',
                kind='box',
                palette=sns.color_palette(palette='pastel'),
                fill=True,
                height=height,
                fliersize=0,
                estimator='median',
                aspect=9.5 / height,
                legend_out=True,
            )

            sns.move_legend(
                g,
                "lower center",
                bbox_to_anchor=(0.41, -0.13),
                ncol=5,
                frameon=True,
            )

            _plot_postprocessing(g=g)
            file_path = join(
                base_output_dir, f'{metric}_{s}_{result_type}_{model_version}.png'
            )
            plt.savefig(file_path, dpi=300, bbox_inches='tight')

            # Disable legend
            g._legend.remove()
            file_path = join(
                base_output_dir,
                f'{metric}_{s}_{result_type}_{model_version}_no_legend.png',
            )
            plt.savefig(file_path, dpi=300, bbox_inches='tight')

            plt.close()

    else:
        average_data = compute_average_score_per_repetition(data=data)
        datasets = [('', data), ('averaged', average_data)]
        for s, d in datasets:
            g = sns.catplot(
                data=d,
                x='Model',
                y=METRIC_NAME_MAP[metric],
                order=MODEL_ORDER,
                hue_order=list_intersection(HUE_ORDER, d['XAI Method'].unique()),
                hue='XAI Method',
                col='Dataset',
                kind='bar',
                palette=sns.color_palette('pastel'),
                height=3,
                estimator='mean',
                # errorbar='sd',
                # err_kws={'linewidth': 2.0},
                aspect=2.0,
                legend_out=True,
            )

            sns.move_legend(
                g,
                "lower center",
                bbox_to_anchor=(0.5, -0.2),
                ncol=5,
                frameon=True,
            )

            _plot_postprocessing(g=g)
            file_path = join(
                base_output_dir, f'{metric}_{s}_{result_type}_{model_version}.png'
            )
            plt.savefig(file_path, dpi=300, bbox_inches='tight')

            # Disable legend
            g._legend.remove()
            file_path = join(
                base_output_dir,
                f'{metric}_{s}_{result_type}_{model_version}_no_legend.png',
            )
            plt.savefig(file_path, dpi=300, bbox_inches='tight')

            plt.close()


def plot_mass_accuracy_reversed(
    data: pd.DataFrame,
    metric: str,
    model_version: str,
    result_type: str,
    base_output_dir: str,
):
    data["mass_accuracy_reversed"] = 1 - data["mass_accuracy"]
    plot_evaluation_results(
        data=data,
        metric="mass_accuracy_reversed",
        model_version=model_version,
        result_type=result_type,
        base_output_dir=base_output_dir,
    )


def plot_evaluation_results_grouped_by_xai_method(
    data: pd.DataFrame,
    metric: str,
    model_version: str,
    result_type: str,
    base_output_dir: str,
) -> None:
    def _plot_postprocessing(g):
        g.set_titles(row_template="Dataset: {row_name}", size=12)
        for k in range(g.axes.shape[0]):
            for j in range(g.axes.shape[1]):
                g.axes[k, j].grid(alpha=0.8, linewidth=0.5)

                if 0 == k and 'top_k_precision' == metric:
                    g.axes[k, j].set_ylabel(
                        f'Average {METRIC_NAME_MAP[metric]}',
                        fontsize=12,
                    )
                else:
                    g.axes[k, j].set_ylabel(
                        f'{METRIC_NAME_MAP[metric]}',
                        fontsize=12,
                    )
                g.axes[k, j].set_xlabel('')
                g.axes[k, j].set_ylim(0, 1)
                g.axes[k, j].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
                # Rotate x-axis labels
                g.axes[k, j].tick_params(axis='x', rotation=45)

                for label in (
                    g.axes[k, j].get_xticklabels() + g.axes[k, j].get_yticklabels()
                ):
                    label.set_fontsize(12)

        # Set legend font size
        g._legend.set_title('XAI Method', prop={'size': 12})
        for t in g._legend.texts:
            t.set_fontsize(12)

    data['mapped_model_name'] = data['model_name'].map(lambda x: MODEL_NAME_MAP[x])
    data['dataset_type'] = data['dataset_type'].map(lambda x: DATASET_NAME_MAP[x])
    data['attribution_method'] = data['attribution_method'].map(
        lambda x: METHOD_NAME_MAP.get(x, x)
    )

    data = data.rename(
        columns={
            "mapped_model_name": "Model",
            "dataset_type": "Dataset",
            "attribution_method": "XAI Method",
            **METRIC_NAME_MAP,
        }
    )

    average_data = compute_average_score_per_repetition(data=data)
    datasets = [('', data), ('averaged', average_data)]

    height = 2.5
    for s, d in datasets:
        g = sns.catplot(
            data=d,
            x='XAI Method',
            y=METRIC_NAME_MAP[metric],
            order=list_intersection(HUE_ORDER, d['XAI Method'].unique()),
            hue_order=MODEL_ORDER,
            row_order=ROW_ORDER,
            hue='Model',
            row='Dataset',
            kind='box',
            palette=sns.color_palette(palette='pastel'),
            fill=True,
            height=height,
            fliersize=0,
            estimator='median',
            aspect=9.5 / height,
            legend_out=True,
        )

        sns.move_legend(
            g,
            "lower center",
            bbox_to_anchor=(0.41, -0.25),
            ncol=5,
            frameon=True,
        )

        _plot_postprocessing(g=g)
        file_path = join(
            base_output_dir, f'{metric}_{s}_{result_type}_{model_version}.png'
        )
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        # Disable legend
        g._legend.remove()
        file_path = join(
            base_output_dir,
            f'{metric}_{s}_{result_type}_{model_version}_no_legend.png',
        )
        plt.savefig(file_path, dpi=300, bbox_inches='tight')

        plt.close()


def plot_model_performance(
    training_history: pd.DataFrame,
    plot_type: str,
    model_version: str,
    base_output_dir: str,
) -> None:
    training_history['mapped_model_name'] = training_history['model_name'].map(
        lambda x: MODEL_NAME_MAP[x]
    )
    training_history['dataset_type'] = training_history['dataset_type'].map(
        lambda x: DATASET_NAME_MAP[x]
    )
    training_history = training_history.rename(
        columns={
            'mapped_model_name': 'Model',
            'dataset_type': 'Dataset',
            'accuracy': 'Accuracy',
            'data_split': 'Data Split',
        }
    )

    g = sns.catplot(
        data=training_history,
        x='Model',
        y='Accuracy',
        hue='Data Split',
        col='Dataset',
        kind='bar',
        order=MODEL_ORDER,
        palette=sns.color_palette(palette='pastel'),
        fill=True,
        linewidth=0.0,
        height=2.5,
        estimator='mean',
        errorbar='sd',
        aspect=2,
        margin_titles=True,
    )
    g.set_titles(col_template="Dataset: {col_name}", size=10)
    for k in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            g.axes[k, j].grid(alpha=0.8, linewidth=0.5)
            g.axes[k, j].set_ylim(0, 1)
            g.axes[k, j].set_yticks(np.arange(start=0.1, step=0.1, stop=1.1))
            g.axes[k, j].set_xlabel('')
            g.axes[k, j].set_ylabel('Accuracy', fontsize=10)

            for label in (
                g.axes[k, j].get_xticklabels() + g.axes[k, j].get_yticklabels()
            ):
                label.set_fontsize(10)

    sns.move_legend(
        g,
        "lower center",
        bbox_to_anchor=(0.47, -0.15),
        ncol=3,
        frameon=True,
    )

    file_path = join(base_output_dir, f'{plot_type}_{model_version}.png')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_most_common_xai_attributions(
    data: pd.DataFrame,
    plot_type: str,
    model_version: str,
    base_output_dir: str,
) -> None:
    data['mapped_model_name'] = data['model_name'].map(lambda x: MODEL_NAME_MAP[x])
    data['dataset_type'] = data['dataset_type'].map(lambda x: DATASET_NAME_MAP[x])

    rows = MODEL_ORDER
    columns = ROW_ORDER
    attribution_methods = np.unique(data['attribution_method'].values)
    ranks = np.unique(data['rank'].values)

    labels = {
        MOST_COMMON_XAI_ATTRIBUTION_PLOT_TYPES[
            'accumulated'
        ]: "Cumulated attribution for female/male",
        MOST_COMMON_XAI_ATTRIBUTION_PLOT_TYPES[
            "tf_idf"
        ]: "TF-IDF normalized cumulated attribution for female/male",
        MOST_COMMON_XAI_ATTRIBUTION_PLOT_TYPES[
            "freq"
        ]: "Frequency normalized cumulated attribution for female/male",
    }

    fig, axs = plt.subplots(
        nrows=len(rows),
        ncols=len(columns),
        sharex=True,
        sharey=True,
        layout='constrained',
        gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
        figsize=(5, 10),
    )

    data = data[data['plot_type'] == plot_type]
    grouped_data = data.groupby(by=['mapped_model_name', 'dataset_type'])

    max_abs_attribution = data['attribution'].abs().max()
    max_abs_attribution = max_abs_attribution + 0.2 * max_abs_attribution

    for k, r in enumerate(rows):
        for j, c in enumerate(columns):

            for keys, df in grouped_data:
                if (r, c) != keys:
                    continue

                for s, (gender, ggdf) in enumerate(df.groupby(by='gender')):
                    ggdf[gender] = (
                        ggdf['attribution'] if 1 == s else (-1) * ggdf['attribution']
                    )

                    g = sns.barplot(
                        data=ggdf,
                        x=gender,
                        y='rank',
                        order=ranks,
                        hue='attribution_method',
                        hue_order=list_intersection(
                            HUE_ORDER, ggdf['attribution_method'].unique()
                        ),
                        orient='y',
                        ax=axs[k, j],
                        width=0.8,
                        native_scale=False,
                        legend=True if 1 == s else False,
                        palette=(
                            sns.color_palette('pastel', len(attribution_methods))
                            if 1 == s
                            else sns.color_palette('muted', len(attribution_methods))
                        ),
                    )
                    start = (
                        0
                        if len(attribution_methods) == len(g.containers)
                        else len(attribution_methods)
                    )
                    for container, (name, mggdf) in zip(
                        g.containers[start:], ggdf.groupby(by='attribution_method')
                    ):
                        g.bar_label(container, labels=mggdf['word'], fontsize=3)

                    axs[k, j].legend(
                        loc='lower right',
                        ncols=1,
                        # fontsize='xx-small',
                        prop={'size': 2},
                    )

                    for label in (
                        axs[k, j].get_xticklabels() + axs[k, j].get_yticklabels()
                    ):
                        label.set_fontsize(4)

                    axs[k, j].set_box_aspect(1)
                    axs[k, j].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

                    axs[k, j].set_xlim(-max_abs_attribution, max_abs_attribution)
                    axs[k, j].set_yticks(ranks, ranks + 1)
                    axs[k, j].set_xlabel(labels[plot_type], fontsize=4)
                    axs[k, j].set_ylabel(ggdf['mapped_model_name'].iloc[0], fontsize=4)

                    axs[k, j].spines['top'].set_linewidth(0.5)
                    axs[k, j].spines['right'].set_linewidth(0.5)
                    axs[k, j].spines['bottom'].set_linewidth(0.5)
                    axs[k, j].spines['left'].set_linewidth(0.5)
                    axs[k, j].grid(linewidth=0.2)
                    if 0 == k:
                        axs[k, j].set_title(
                            f'Dataset: {ggdf["dataset_type"].iloc[0]}',
                            fontsize=4,
                        )

    file_path = join(base_output_dir, f'{plot_type}_{model_version}.png')
    logger.info(file_path)
    plt.savefig(file_path, dpi=300)
    plt.close()


def create_evaluation_plots(base_output_dir: str, config: dict) -> None:
    artifacts_dir = generate_artifacts_dir(config=config)
    evaluation_dir = generate_evaluation_dir(config=config)
    file_path = join(
        artifacts_dir, evaluation_dir, config['evaluation']['evaluation_records']
    )
    evaluation_results: EvaluationResult = load_pickle(file_path)
    xai_visualization_methods = dict(
        roc_auc=plot_evaluation_results,
        precision_recall_auc=plot_evaluation_results,
        avg_precision=plot_evaluation_results,
        precision_specificity=plot_evaluation_results,
        top_k_precision=plot_evaluation_results,
        mass_accuracy=plot_evaluation_results,
        mass_accuracy_method_grouped=plot_evaluation_results_grouped_by_xai_method,
        mass_accuracy_reversed=plot_mass_accuracy_reversed,
    )

    plot_types = config['visualization']['visualizations']['evaluation']
    for plot_type in plot_types:
        for result_df, result_type in [
            (evaluation_results.xai_results_all, "filter_all"),
            (evaluation_results.xai_results_correct, "filter_correct"),
        ]:
            result_df = pd.DataFrame(result_df)
            if len(result_df) == 0:
                # E.g. for sentiment dataset we have no ground truth labels and therefore
                # the evaluation results are empty for the correct filter
                continue

            # Filter out methods not listed in config
            methods = result_df['attribution_method'].unique()
            for method in methods:
                if method not in config["xai"]["methods"]:
                    result_df = result_df[result_df['attribution_method'] != method]

            for model_version, xai_group in result_df.groupby("model_version"):
                logger.info(f'Type of plot: {plot_type}')
                v = xai_visualization_methods.get(plot_type, None)
                if v is not None:
                    v(xai_group, plot_type, model_version, result_type, base_output_dir)


def create_model_performance_plots(base_output_dir: str, config: dict) -> None:
    artifacts_dir = generate_artifacts_dir(config=config)

    def load_training_history(
        records: list, eval_results: EvaluationResult
    ) -> pd.DataFrame:
        data_dict = dict(
            dataset_type=list(),
            model_name=list(),
            model_version=list(),
            accuracy=list(),
            data_split=list(),
        )

        for record in records:
            # history_path = join(*record[-1].split('/')[2:])
            history_path = record[-1]
            model_info = record[1]

            training_history = load_pickle(file_path=join(artifacts_dir, history_path))
            data_dict['dataset_type'] += [record[0].split('_')[-1]]
            data_dict['model_name'] += [record[1]['model_name']]
            data_dict['model_version'] += [model_info["save_version"].value]
            data_dict['accuracy'] += [training_history['train_acc'][-1]]
            data_dict['data_split'] += ['training']

            data_dict['dataset_type'] += [record[0].split('_')[-1]]
            data_dict['model_name'] += [record[1]['model_name']]
            data_dict['model_version'] += [model_info["save_version"].value]
            data_dict['accuracy'] += [training_history['val_acc'][-1]]
            data_dict['data_split'] += ['validation']

        for entry in eval_results.model_results:
            data_dict['dataset_type'] += [entry.dataset_type.split("_")[-1]]
            data_dict['model_name'] += [entry.model_name]
            data_dict['model_version'] += [entry.model_version]
            data_dict['accuracy'] += [entry.accuracy]
            data_dict['data_split'] += ["test"]

        return pd.DataFrame(data_dict)

    training_dir = generate_training_dir(config=config)
    artifacts_dir = generate_artifacts_dir(config=config)
    file_path = join(
        artifacts_dir, training_dir, config['training']['training_records']
    )
    training_records = load_pickle(file_path=file_path)

    evaluation_dir = generate_evaluation_dir(config=config)
    file_path = join(
        artifacts_dir, evaluation_dir, config['evaluation']['evaluation_records']
    )
    evaluation_results: EvaluationResult = load_pickle(file_path)

    history = load_training_history(
        records=training_records, eval_results=evaluation_results
    )

    history_visualization_methods = dict(
        model_performance=plot_model_performance,
    )

    plot_types = config['visualization']['visualizations']['model']
    for plot_type in plot_types:
        for model_version, group in history.groupby("model_version"):
            logger.info(f'Type of plot: {plot_type} for model_version {model_version}')
            v = history_visualization_methods.get(plot_type, None)
            if v is not None:
                v(group, plot_type, model_version, base_output_dir)


def create_legend_plot(base_output_dir: str, model: str, data: pd.DataFrame):
    # Save only plot legend as separte figure to be appended in HTML file at the bottom
    folder_path = join(base_output_dir, f"{model}_xai_attributions_per_word")
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    file_path_legend_plot = join(folder_path, f'plot_legend.png')

    h = sns.barplot(
        data=data,
        x="method",
        y="attribution",
        hue="method",
        hue_order=list_intersection(HUE_ORDER, data['method'].unique()),
        order=list_intersection(HUE_ORDER, data['method'].unique()),
        palette=sns.color_palette('pastel'),
        # width=0.8
    )

    # Get the unique colors of the bars
    colors = [p.get_facecolor() for p in h.patches]
    plt.close()
    # Create custom legend
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
        for color in colors
    ]

    # Create a new figure for the legend
    fig_legend = plt.figure(figsize=(5, 5))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.legend(
        title='XAI Method',
        handles=legend_patches,
        labels=list_intersection(HUE_ORDER, data['method'].unique()),
        loc='center',
        ncol=len(legend_patches) / 2,
        frameon=True,
    )

    ax_legend.axis('off')  # Hide the axes
    # Save the legend as a figure
    fig_legend.savefig(file_path_legend_plot, bbox_inches='tight', dpi=300)
    plt.close(fig_legend)

    # GPT4-generated code to remove the white borders around legend image
    # Makes border handling with the respect to the whole html file easier and
    # Ensures a tighter layout
    from PIL import Image

    def trim_whitespace(image_path):
        with Image.open(image_path) as img:
            # Convert to a NumPy array for image processing
            img_array = np.array(img)
            # Find non-white pixels
            non_white_pix = np.where(img_array < 255)
            # Get the bounding box of non-white pixels
            bbox = [
                np.min(non_white_pix[1]),
                np.min(non_white_pix[0]),
                np.max(non_white_pix[1]),
                np.max(non_white_pix[0]),
            ]
            # Crop the image according to the bounding box
            trimmed_img = img.crop(bbox)
            # Save the trimmed image
            trimmed_img.save(file_path_legend_plot)

    # Call the function to trim whitespace
    trim_whitespace(file_path_legend_plot)

    return file_path_legend_plot


def create_single_word_plot(
    data: pd.DataFrame,
    model: str,
    word_idx: int,
    sentence: list,
    base_output_dir: str,
):
    # Set the size of the figure (width, height)
    plt.figure(figsize=(10, 10))

    # Barplots for each word
    g = sns.barplot(
        data=data,
        x="method",
        y="attribution",
        hue="method",
        hue_order=list_intersection(HUE_ORDER, data['method'].unique()),
        order=list_intersection(HUE_ORDER, data['method'].unique()),
        width=0.8,
        palette=sns.color_palette('pastel'),
        legend=False,
    )

    # Set the border color and width
    for bar in g.patches:
        bar.set_edgecolor('black')  # Set the border color
        bar.set_linewidth(2)  # Set the border width

    g.set_ylabel("")
    g.set_xlabel("")

    sns.despine(left=True, bottom=True)
    g.set_yticklabels([])
    g.tick_params(left=False)

    g.set_xticklabels([])
    plt.yticks(np.arange(0, 1.1, 0.1))

    folder_path = join(base_output_dir, f"{model}_xai_attributions_per_word")
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    file_path = join(
        folder_path, f'{str(word_idx)}_attributions_word_{sentence[word_idx]}.png'
    )

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

    return file_path


def create_xai_sentence_html_plots(
    data: pd.DataFrame, plot_type: str, base_output_dir: str
) -> None:

    data['attribution_method'] = data['attribution_method'].map(
        lambda x: METHOD_NAME_MAP.get(x, x)
    )

    sentence_idx = 122  # 1179
    repetition_number = 0
    target = 0  # 0 female, 1 male
    model_version = SaveVersion.best.value
    dataset_type = DatasetKeys.gender_all.value

    base_output_dir = join(base_output_dir, f'xai_sentence_plots_{sentence_idx}')

    df_explanations_sentence_different_models = data[
        (data["sentence_idx"] == sentence_idx)
        & (data["target"] == target)
        & (data["model_version"] == model_version)
        & (data["dataset_type"] == dataset_type)
        & (data["model_repetition_number"] == repetition_number)
    ]

    pre_trained_models = [
        'bert_only_classification',
        'bert_randomly_init_embedding_classification',
        'bert_only_embedding_classification',
        'bert_all',
        'one_layer_attention_classification',
    ]

    model_image_paths = []
    image_model_captions = []
    for model in tqdm(
        pre_trained_models, desc="Generating images for different models."
    ):

        # Generate plot images
        df_model = df_explanations_sentence_different_models[
            df_explanations_sentence_different_models['model_name'] == model
        ]

        sentence = df_model['sentence'].iloc[0]
        xai_methods_per_sentence = df_model['attribution_method']
        attribution_scores_per_sentence = df_model['attribution']
        ground_truth_per_sentence = df_model['ground_truth'].iloc[0]

        sentences_w_ground_truths = list(zip(sentence, ground_truth_per_sentence))

        # Add caption
        model_caption = ["" for _ in range(len(sentences_w_ground_truths))]
        model_caption[0] = MODEL_NAME_HTML_MAP[model]
        image_model_captions.append(model_caption)

        image_paths = []
        for word_idx in range(len(sentence)):
            attribution_scores_per_word = []
            xai_methods_per_word = []

            for method in range(len(attribution_scores_per_sentence)):
                attribution_scores_per_word.append(
                    attribution_scores_per_sentence.iloc[method][word_idx]
                )
                xai_methods_per_word.append(xai_methods_per_sentence.iloc[method])

            data = pd.DataFrame(
                {
                    'method': xai_methods_per_word,
                    'attribution': attribution_scores_per_word,
                }
            )

            file_path_legend_plot = create_legend_plot(
                base_output_dir=base_output_dir, model=model, data=data
            )

            file_path = create_single_word_plot(
                data=data,
                model=model,
                word_idx=word_idx,
                sentence=sentence,
                base_output_dir=base_output_dir,
            )

            image_paths.append(file_path)

        model_image_paths.append(image_paths)

    # GPT4-generated code
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Concatenated Images</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/dejavu-sans@1.0.0/css/dejavu-sans.min.css">
        <style>
            body {
                font-family: 'DejaVu Sans', sans-serif;
            }
            .image-container{
                display: flex;
                flex-direction: row;
                justify-content: flex-start;
                align-items: center;
            }
            .legend-plot{
                display: flex; /* Use flexbox to center the content */
                justify-content: left; /* Center horizontally */
                align-items: left; /* Center vertically */
                margin-top: 20px; /* Add some space above the vertical image row */
            }
            .image-box {
                margin-right: 0px; /* Adjust spacing between image-text blocks */
            }
            .image-box img {
                max-width: 120px; /* Set a maximum width for each image */
                max-height: 120px; /* Set a maximum height for each image */
                object-fit: contain; /* Ensure the aspect ratio of images is maintained */
                display: block; /* Makes the image a block-level element */
                margin-bottom: 0px; /* Spacing between image and text */
            }
            .image-text {
                text-align: center; /* Center-aligns the text below the image */
            }
            .highlight {
                background-color: lightgreen; /* Highlight color */
                border-radius: 4px;
            }
            .legend-plot img {
                max-width: 1250px; /* Adjust max width as needed */
                object-fit: contain;
                margin-top: 10px;
            }
            .image-with-caption {
                display: flex;
                align-items: center; /* Vertically center the flex items */
            }
            .text-container {
                width: 100px; /* Fixed width for the text container */
                transform: rotate(-90deg);
                margin-right: -25px; /* Consistent margin to the right of the text */
                margin-left: -25px; /* Consistent margin to the right of the text */
            }
        </style>
    </head>
    <body>
        <div class="image-container">
    '''

    def path_to_url(file_path: str):
        return "/artifacts/" + file_path.split("/artifacts/")[1]

    image_model_captions_zipped = zip(*image_model_captions)
    image_model_captions_zipped = [list(group) for group in image_model_captions_zipped]

    model_image_paths_zipped = [list(group) for group in zip(*model_image_paths)]

    for index, (model_name_caption, img_path, (text, highlight)) in enumerate(
        zip(
            image_model_captions_zipped,
            model_image_paths_zipped,
            sentences_w_ground_truths,
        )
    ):
        if (
            exists(img_path[0])
            and exists(img_path[1])
            and exists(img_path[2])
            and exists(img_path[3])
            and exists(img_path[4])
        ):
            if index == 0:
                highlight_class = 'highlight' if highlight else ''
                html_content += f'''
                <div class="image-box">
                    <div class="image-with-caption">
                        <div class="text-container">{model_name_caption[0]}</div>
                            <div class="image-model-one">
                                <img src="{path_to_url(img_path[0])}" alt="Image">
                            <div class="image-text {highlight_class}">{text}</div>
                        </div>
                    </div>
                    
                    <div class="image-with-caption">
                        <div class="text-container">{model_name_caption[1]}</div>
                            <div class="image-model-two">
                                <img src="{path_to_url(img_path[1])}" alt="Image">
                            <div class="image-text {highlight_class}">{text}</div>
                        </div>
                    </div>
                                    
                    <div class="image-with-caption">
                        <div class="text-container">{model_name_caption[2]}</div>
                            <div class="image-model-two">
                                <img src="{path_to_url(img_path[2])}" alt="Image">
                            <div class="image-text {highlight_class}">{text}</div>
                        </div>
                    </div>

                    <div class="image-with-caption">
                        <div class="text-container">{model_name_caption[3]}</div>
                            <div class="image-model-two">
                                <img src="{path_to_url(img_path[3])}" alt="Image">
                            <div class="image-text {highlight_class}">{text}</div>
                        </div>
                    </div>

                    <div class="image-with-caption">
                        <div class="text-container">{model_name_caption[4]}</div>
                            <div class="image-model-two">
                                <img src="{path_to_url(img_path[4])}" alt="Image">
                            <div class="image-text {highlight_class}">{text}</div>
                        </div>
                    </div>
                </div>
                '''
            else:
                highlight_class = 'highlight' if highlight else ''
                html_content += f'''
                <div class="image-box">
                    <div class="image-model-one">
                        <img src="{path_to_url(img_path[0])}" alt="Image">
                    </div>
                    <div class="image-text {highlight_class}">{text}</div>

                    <div class="image-model-two">
                        <img src="{path_to_url(img_path[1])}" alt="Image">
                    </div>
                    <div class="image-text {highlight_class}">{text}</div>
                                    
                    <div class="image-model-three">
                        <img src="{path_to_url(img_path[2])}" alt="Image">
                    </div>
                    <div class="image-text {highlight_class}">{text}</div>
                                    
                    <div class="image-model-fourth">
                        <img src="{path_to_url(img_path[3])}" alt="Image">
                    </div>
                    <div class="image-text {highlight_class}">{text}</div>

                    <div class="image-model-fourth">
                        <img src="{path_to_url(img_path[4])}" alt="Image">
                    </div>
                    <div class="image-text {highlight_class}">{text}</div>
                </div>
                '''
        else:
            print(f"Warning: Image Path not found.")

    if exists(file_path_legend_plot):
        html_content += f'''
            </div> <!-- Closing image-container -->
            <!-- New container for the vertical image -->
            <div class="legend-plot">
                <img src="{path_to_url(file_path_legend_plot)}" alt="Legend Plot">
            </div>
        </body>
        </html>
        '''
    else:
        print(f"Warning: Image {file_path_legend_plot} not found.")

    file_path = join(
        base_output_dir,
        f'{sentence_idx}_{dataset_type}_{target}_{model_version}_models_xai_sentence_html_plot.html',
    )
    with open(file_path, 'w') as file:
        file.write(html_content)


def get_tfidf_weights(
    df: pd.DataFrame,
):
    # Create a vocab of the words in the dataset. Needed as we split the sentence based
    # on the tokenizer and not based on whitespaces.
    vocab = set()
    for i, row in df.iterrows():
        vocab.update(row['sentence'])

    vectorizer = TfidfVectorizer(vocabulary=vocab, lowercase=False)
    X = vectorizer.fit_transform(df['sentence'].apply(lambda x: ' '.join(x)))

    return X, vectorizer


def calculate_most_common_xai_attributions(value):
    data_dict = dict(
        gender=list(),
        dataset_type=list(),
        model_name=list(),
        model_version=list(),
        word=list(),
        attribution=list(),
        attribution_method=list(),
        rank=list(),
        plot_type=list(),
    )

    keys, df = value
    word_frequencies = dict()
    accumulated_attributions = dict()
    freq_normalized_attributions = dict()
    tf_idf_normalized_attributions = dict()

    tf_idf_weights, vectorizer = get_tfidf_weights(df=df)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        for k, word in enumerate(row['sentence']):
            if word not in accumulated_attributions:
                accumulated_attributions[word] = row['attribution'][k]
                tf_idf_normalized_attributions[word] = (
                    tf_idf_weights[row_idx, vectorizer.vocabulary_[word]]
                    * row['attribution'][k]
                )
                word_frequencies[word] = 1
            else:
                accumulated_attributions[word] += row['attribution'][k]
                tf_idf_normalized_attributions[word] += (
                    tf_idf_weights[row_idx, vectorizer.vocabulary_[word]]
                    * row['attribution'][k]
                )
                word_frequencies[word] += 1

    for word in accumulated_attributions:
        r = accumulated_attributions[word] / word_frequencies[word]
        freq_normalized_attributions[word] = r

    def add_to_data(word_counter: Counter, plot_type: str):
        for i, (word, attribution) in enumerate(word_counter):
            data_dict['model_name'] += [keys[0]]
            data_dict['dataset_type'] += [keys[1]]
            data_dict['gender'] += [GENDER[keys[2]]]
            data_dict['attribution_method'] += [keys[3]]
            data_dict['model_version'] += [keys[4]]
            data_dict['word'] += [word]
            data_dict['attribution'] += [attribution]
            data_dict['rank'] += [i]
            data_dict['plot_type'] += [plot_type]

    add_to_data(
        Counter(accumulated_attributions).most_common(n=5),
        MOST_COMMON_XAI_ATTRIBUTION_PLOT_TYPES['accumulated'],
    )
    add_to_data(
        Counter(tf_idf_normalized_attributions).most_common(n=5),
        MOST_COMMON_XAI_ATTRIBUTION_PLOT_TYPES['tf_idf'],
    )
    add_to_data(
        Counter(freq_normalized_attributions).most_common(n=5),
        MOST_COMMON_XAI_ATTRIBUTION_PLOT_TYPES['freq'],
    )

    return pd.DataFrame(data_dict)


def create_dataset_for_xai_plot(
    plot_type: str, xai_records: list
) -> pd.DataFrame | DataFrameGroupBy:
    output = None
    if 'most_common_xai_attributions' in plot_type:
        data = pd.DataFrame(xai_records)
        grouped_data = data.groupby(
            by=[
                'model_name',
                'dataset_type',
                'target',
                'attribution_method',
                'model_version',
            ]
        )

        result = process_map(
            calculate_most_common_xai_attributions,
            grouped_data,
            max_workers=8,
            desc="Preparing most common xai attributions",
        )
        return pd.concat(result)

    elif 'sentence_html_plot' == plot_type:
        output = pd.DataFrame(xai_records)
    return output


def load_xai_records(config: dict) -> pd.DataFrame:
    xai_dir = generate_xai_dir(config=config)
    artifacts_dir = generate_artifacts_dir(config=config)
    file_path = join(artifacts_dir, xai_dir, config['xai']['xai_records'])
    paths_to_xai_records = load_pickle(file_path=file_path)
    data_list = list()
    for p in tqdm(paths_to_xai_records, desc="Loading xai records."):
        results = load_pickle(file_path=join(artifacts_dir, p))
        for xai_records in results:
            xai_records.model_version = xai_records.model_version.value
            data_list += [asdict(xai_records)]

    df = pd.DataFrame(data_list)

    methods = df['attribution_method'].unique()
    for method in methods:
        if method not in config["xai"]["methods"]:
            df = df[df['attribution_method'] != method]

    return df


def create_xai_plots(base_output_dir: str, config: dict) -> None:
    xai_records = load_xai_records(config=config)

    visualization_methods = dict(
        most_common_xai_attributions=plot_most_common_xai_attributions,
        most_common_xai_attributions_tf_idf=plot_most_common_xai_attributions,
        most_common_xai_attributions_freq=plot_most_common_xai_attributions,
        sentence_html_plot=create_xai_sentence_html_plots,
    )

    most_common_data = None
    plot_types = config['visualization']['visualizations']['xai']
    for plot_type in plot_types:
        logger.info(f'Type of plot: {plot_type}')
        v = visualization_methods.get(plot_type, None)
        base_output_dir = (
            join(generate_project_dir(config=config), base_output_dir)
            if plot_type == 'sentence_html_plot'
            else base_output_dir
        )
        if v is None:
            continue

        if 'sentence_html_plot' == plot_type:
            data = create_dataset_for_xai_plot(
                plot_type=plot_type, xai_records=xai_records
            )
            v(data, plot_type, base_output_dir)
        else:
            if most_common_data is None:
                # Cache results for different kinds e.g. freq, tf-idf, accumulated
                most_common_data = create_dataset_for_xai_plot(
                    plot_type=plot_type, xai_records=xai_records
                )

            for model_version, group in most_common_data.groupby("model_version"):
                v(group, plot_type, model_version, base_output_dir)


def calculate_correlation_between_words_and_labels(
    sentences: list, labels: list, mode: str = 'tfidf'
) -> Tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer() if 'tfidf' == mode else None
    x = vectorizer.fit_transform(sentences).todense()
    y = np.array(labels)
    xy = np.concatenate((x, y[:, np.newaxis]), axis=1)
    correlation_xy = np.corrcoef(xy, rowvar=False)[-1, :-1]

    sorted_indices = np.argsort(correlation_xy)
    sorted_feature_names = np.array(vectorizer.get_feature_names_out())[sorted_indices]
    sorted_correlations = correlation_xy[sorted_indices]

    return sorted_correlations, sorted_feature_names


def plot_correlation_between_words_and_labels(
    data: dict, plot_type: str, output_dir: str, config: dict
) -> None:
    dataset_types = list(data.keys())
    top_k = 10
    ranks = np.arange(start=0, stop=top_k, step=1)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(dataset_types),
        sharex=True,
        sharey=True,
        gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
        figsize=(4, 2),
    )

    for k, c in enumerate(dataset_types):
        df = data[c]
        sentences = df['sentence'].map(lambda x: ' '.join(map(str, x))).to_list()
        y = df['target'].to_list()
        correlations, words = calculate_correlation_between_words_and_labels(
            sentences=sentences,
            labels=y,
            mode='tfidf',
        )

        for g in ['female', 'male']:
            if 'female' == g:
                topk_correlations = np.abs(correlations[:top_k])
                topk_words = words[:top_k]
                colors = sns.color_palette('muted', len(topk_words))
            else:
                topk_correlations = (-1) * np.abs(correlations[::-1][:top_k])
                topk_words = words[::-1][:top_k]
                colors = sns.color_palette('pastel', len(topk_words))

            plot_df = pd.DataFrame(
                {'correlations': topk_correlations, 'words': topk_words, 'ranks': ranks}
            )

            g = sns.barplot(
                data=plot_df,
                x='correlations',
                y='ranks',
                order=ranks,
                hue='words',
                orient='y',
                ax=axs[k],
                width=0.8,
                native_scale=False,
                palette=colors,
                legend=False,
            )

            start = 0 if len(topk_words) == len(g.containers) else len(topk_words)
            for container, word in zip(g.containers[start:], topk_words):
                g.bar_label(container, labels=[word], fontsize=4)

            for label in axs[k].get_xticklabels() + axs[k].get_yticklabels():
                label.set_fontsize(4)
            axs[k].set_box_aspect(1)
            axs[k].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            # axs[k].set_xticks([-1, -0.5, 0], [0, 0.5, 1])
            axs[k].set_xticks([-1, -0.5, 0, 0.5, 1], [1, 0.5, 0, 0.5, 1])
            axs[k].set_yticks(ranks, ranks + 1)
            axs[k].set_xlabel(
                '$\mathrm{Corr}(x_{TFidf}, y_{male})$ vs. $\mathrm{Corr}(x_{TFidf}, y_{female})$',
                fontsize=4,
            )
            axs[k].xaxis.set_label_coords(0.5, -0.11)
            axs[k].set_ylabel('Rank', fontsize=4)
            axs[k].spines['top'].set_linewidth(0.5)
            axs[k].spines['right'].set_linewidth(0.5)
            axs[k].spines['bottom'].set_linewidth(0.5)
            axs[k].spines['left'].set_linewidth(0.5)
            axs[k].grid(linewidth=0.1)
            axs[k].set_title(
                f'Dataset: {DATASET_NAME_MAP[c]}',
                fontsize=4,
            )

    # fig.suptitle('Correlation between Tfidf represented words and labels', fontsize=4)
    file_path = join(output_dir, f'{plot_type}.png')
    logger.info(file_path)
    # plt.tight_layout()
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_data_plots(base_output_dir: str, config: dict) -> None:
    filename_all = join(
        generate_data_dir(config), DatasetKeys.gender_all.value, "test.jsonl"
    )
    filename_subj = join(
        generate_data_dir(config), DatasetKeys.gender_subj.value, "test.jsonl"
    )

    dataset_all = load_jsonl_as_df(filename_all)
    dataset_subject = load_jsonl_as_df(filename_subj)

    data = dict(gender_all=dataset_all, gender_subj=dataset_subject)
    visualization_methods = dict(
        correlation_plot=plot_correlation_between_words_and_labels,
    )

    plot_types = config['visualization']['visualizations']['data']
    for plot_type in plot_types:
        logger.info(f'Type of plot: {plot_type}')
        v = visualization_methods.get(plot_type, None)
        if v is None:
            continue
        v(data, plot_type, base_output_dir, config)


def model_ds_axs(
    data: pd.DataFrame, font_size=None, figsize: tuple = (10, 10), **kwargs
):
    """
    Helper function that creates a grid of subplots for each model and dataset type.
    and groups the data accordingly.
    """
    data["model_name"] = data["model_name"].map(MODEL_NAME_MAP)
    data["dataset_type"] = data["dataset_type"].map(DATASET_NAME_MAP)
    pad = 5
    _, axs = plt.subplots(
        nrows=len(data["model_name"].unique()),
        ncols=len(data["dataset_type"].unique()),
        figsize=figsize,
        **kwargs,
    )

    plots = []
    for model_idx, model_name in enumerate(MODEL_ORDER):
        for ds_idx, dataset_type in enumerate(ROW_ORDER):
            group = data[
                (data["model_name"] == model_name)
                & (data["dataset_type"] == dataset_type)
            ]

            ax = axs[model_idx][ds_idx]

            # Set title and labels
            if model_idx == 0:
                ax.set_title(f"Dataset: {dataset_type}", fontsize=font_size)

            if ds_idx == 0:
                ax.annotate(
                    model_name,
                    xy=(0, 0.5),
                    xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label,
                    textcoords='offset points',
                    ha='right',
                    va='center',
                    rotation=90,
                    fontsize=font_size,
                )

            plots.append((ax, model_name, dataset_type, group))

    return plots


def plot_prediction_prob_diff(
    data: pd.DataFrame,
    plot_type: str,
    model_version: str,
    base_output_dir: str,
    config: dict,
) -> None:
    """
    The plot shows the differences in the probability of positive sentiment between
    the male and female version of the same sentence.

    A positive value in the difference means that the model assigns a higher
    probability to a positive sentiment for female sentences and a negative value
    means that the model assigns a higher positive sentiment probability to male sentences on average.
    """

    plots = model_ds_axs(data, figsize=(15, 10), sharex=True, sharey=True)

    plot_diffs = []
    for _, _, _, grouped in plots:
        # Prepare data for plot
        diffs = []
        for _, group in grouped.groupby(by=["sentence_idx", "model_repetition_number"]):
            female_positiv_prob = group[group["target"] == 0].iloc[0][
                "pred_probabilities"
            ][1]
            male_positive_prob = group[group["target"] == 1].iloc[0][
                "pred_probabilities"
            ][1]
            diffs += [female_positiv_prob - male_positive_prob]

        plot_diffs.append(diffs)

    max_diff = np.max(np.abs(diffs))
    for diffs, (ax, _, _, _) in zip(plot_diffs, plots):
        # Plot histograms
        ax.set_xlabel("Probability")
        ax.set_ylabel("Counts")
        ax.hist(diffs, bins=50, range=(-max_diff, max_diff))
        ax.legend()

    plt.suptitle("Difference in probabilities of positive sentiment\n female - male")
    file_path = join(base_output_dir, f'{plot_type}_{model_version}.png')
    logger.info(file_path)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_positive(
    data: pd.DataFrame,
    plot_type: str,
    model_version: str,
    base_output_dir: str,
    config: dict,
) -> None:
    """
    The plot shows the distribution of the probability of positive sentiment for
    male and female sentences.
    """

    for ax, _, _, grouped in model_ds_axs(
        data, figsize=(15, 10), sharex=True, sharey=True
    ):
        # Prepare data for plot
        female_positiv_probs = [
            pred[1] for pred in grouped[grouped["target"] == 0]["pred_probabilities"]
        ]
        male_positive_probs = [
            pred[1] for pred in grouped[grouped["target"] == 1]["pred_probabilities"]
        ]

        # Plot histograms
        ax.set_xlabel("Probability")
        ax.set_ylabel("Counts")
        ax.set_xlim(0, 1)
        ax.hist(female_positiv_probs, bins=50, label="Female positive", alpha=0.5)
        ax.hist(male_positive_probs, bins=50, label="Male positive", alpha=0.5)
        ax.legend()

    plt.suptitle("Probability of positive sentiment")
    file_path = join(base_output_dir, f'{plot_type}_{model_version}.png')
    logger.info(file_path)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prediction_diff(
    data: pd.DataFrame,
    plot_type: str,
    model_version: str,
    base_output_dir: str,
    config: dict,
) -> None:
    """
    The plot shows how the predicted labels positive/negative differ for the
    male and the female version of the same sentence.

    e.g. if the male version of the same sentence is predicted as positive and the
    female version as negative or vice versa.
    """

    _, axs = plt.subplots(
        ncols=len(data["dataset_type"].unique()),
        figsize=(6, 3),
        sharex=True,
        sharey=True,
    )

    sections = [
        "Same sentiment",
        'Male positive, female negative',
        'Female positive, male negative',
    ]
    for ds_idx, (dataset_type, group) in enumerate(data.groupby(by="dataset_type")):
        ax = axs[ds_idx]
        ax.set_title(f"Dataset: {DATASET_NAME_MAP[dataset_type]}")

        plot_data = []
        for model_name, group_model in group.groupby(by="model_name"):
            # Prepare data for plot
            same = 0
            male_pos_fem_neg = 0
            fem_pos_male_neg = 0
            for _, group in group_model.groupby(
                by=["sentence_idx", "model_repetition_number"]
            ):
                female_pred = np.argmax(
                    group[group["target"] == 0].iloc[0]["pred_probabilities"]
                )
                male_pred = np.argmax(
                    group[group["target"] == 1].iloc[0]["pred_probabilities"]
                )

                if female_pred == male_pred:
                    same += 1
                elif male_pred == 1:
                    male_pos_fem_neg += 1
                else:
                    fem_pos_male_neg += 1

            total = same + male_pos_fem_neg + fem_pos_male_neg

            plot_data.append(
                {
                    'model': model_name,
                    'section': sections[0],
                    'count': (same / total) * 100,
                }
            )

            plot_data.append(
                {
                    'model': model_name,
                    'section': sections[1],
                    'count': (male_pos_fem_neg / total) * 100,
                }
            )

            plot_data.append(
                {
                    'model': model_name,
                    'section': sections[2],
                    'count': (fem_pos_male_neg / total) * 100,
                }
            )

        plot_data = pd.DataFrame(plot_data)
        plot_data['model'] = plot_data['model'].map(MODEL_NAME_MAP)

        sns.barplot(
            x="count",
            y="section",
            order=sections,
            hue="model",
            hue_order=list_intersection(MODEL_ORDER, plot_data['model'].unique()),
            palette=sns.color_palette("pastel"),
            orient="y",
            data=plot_data,
            ax=ax,
            legend=ds_idx == 0,
        )

        # Disable y-axis labels
        ax.set(ylabel=None, xlabel='% of sentences')

        if ds_idx == 0:
            # Move legend outside of plot
            ax.legend(
                loc='lower center', bbox_to_anchor=(0.75, -0.5), title="Model", ncol=5
            )

    file_path = join(base_output_dir, f'{plot_type}_{model_version}.png')
    logger.info(file_path)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_sentence_wise_attribution_diff(data: pd.DataFrame) -> None:
    """
    Calculates the difference in attributions for each word in a sentence between
    the male and female version of the sentence.
    """
    attribution_diff = dict()

    def add_diff(word, diff):
        if word not in attribution_diff:
            attribution_diff[word] = []
        attribution_diff[word].append(diff)

    for _, group in tqdm(data.groupby(by=["sentence_idx", "model_repetition_number"])):
        assert len(group) == 2

        female = group[group["target"] == 0].iloc[0]
        male = group[group["target"] == 1].iloc[0]

        for female_word, male_word, female_attr, male_attr in zip(
            female["sentence"],
            male["sentence"],
            female["attribution"],
            male["attribution"],
        ):
            diff = np.abs(female_attr - male_attr)
            if female_word == male_word:
                add_diff(female_word, diff)
            else:
                add_diff(
                    f"{female_word.lower()} / {male_word.lower()}",
                    diff,
                )

    # Calculate mean attribution difference for each word
    for word in attribution_diff:
        attribution_diff[word] = np.mean(attribution_diff[word])

    return attribution_diff


def plot_sentence_wise_attribution_diff(
    data: pd.DataFrame,
    plot_type: str,
    model_version: str,
    base_output_dir: str,
    config: dict,
) -> None:
    """
    The plot shows the top k words with the highest attribution difference of words
    for the male and female version of the same sentence.
    """
    top_k = 5

    for ax, _, _, grouped in model_ds_axs(
        data, font_size=4, figsize=(5, 12), sharex=True
    ):
        # Prepare word data for plot
        dfs = []
        for method, grouped_method in grouped.groupby(by="attribution_method"):
            attribution_diff = calculate_sentence_wise_attribution_diff(grouped_method)
            method_df = pd.DataFrame(
                Counter(attribution_diff).most_common(n=5),
                columns=["word", "abs_difference"],
            )
            method_df["rank"] = np.arange(1, top_k + 1)
            method_df["method"] = [method] * top_k
            dfs.append(method_df)

        df = pd.concat(dfs)

        # Define order of ranks and hue order / method order
        ranks = np.unique(df['rank'].values)
        mean_attr_method = df[df["rank"] == 1].groupby(by="method").mean()
        mean_attr_method.sort_values(by="abs_difference", ascending=False, inplace=True)

        g = sns.barplot(
            x="abs_difference",
            y="rank",
            order=ranks,
            hue="method",
            hue_order=list_intersection(HUE_ORDER, df['method'].unique()),
            orient="y",
            data=df,
            ax=ax,
            width=0.8,
            native_scale=False,
            legend=True,
        )

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(4)

        ax.set(ylabel=None, xlim=(0, 1))
        ax.set_xlabel("Absolute attribution difference", fontsize=4)

        ax.legend(
            loc='lower right',
            ncol=1,
            prop={'size': 2},
        )

        # Add word labels to bars
        for container, (name, mggdf) in zip(g.containers, df.groupby(by='method')):
            g.bar_label(container, labels=mggdf['word'], fontsize=3, padding=3)

    file_path = join(base_output_dir, f'{plot_type}_{model_version}.png')
    logger.info(file_path)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_prediction_plots(base_output_dir: str, config: dict) -> None:
    xai_results = load_xai_records(config=config)
    xai_results = pd.DataFrame(xai_results)

    visualization_methods = dict(
        prediction_positive=plot_prediction_positive,
        prediction_prob_diff=plot_prediction_prob_diff,
        prediction_diff=plot_prediction_diff,
        sentence_wise_attribution_diff=plot_sentence_wise_attribution_diff,
    )

    plot_types = config['visualization']['visualizations']['prediction']
    for plot_type in plot_types:
        logger.info(f'Type of plot: {plot_type}')
        v = visualization_methods.get(plot_type, None)
        if v is None:
            continue
        for model_version, group in xai_results.groupby("model_version"):
            v(group, plot_type, model_version, base_output_dir, config)


def visualize_results(base_output_dir: str, config: dict) -> None:
    for visualization, _ in config['visualization']['visualizations'].items():
        v = VISUALIZATIONS.get(visualization, None)
        if v is not None:
            v(base_output_dir, config)


VISUALIZATIONS = dict(
    data=create_data_plots,
    xai=create_xai_plots,
    evaluation=create_evaluation_plots,
    model=create_model_performance_plots,
    prediction=create_prediction_plots,
    gender_difference=create_gender_difference_plots,
)


def main(config: dict) -> None:
    rc('text', usetex=True)
    output_dir = generate_visualization_dir(config=config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    visualize_results(base_output_dir=output_dir, config=config)


if __name__ == "__main__":
    main(config={})
