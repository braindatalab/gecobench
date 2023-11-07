from copy import deepcopy
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt

from utils import generate_visualization_dir, generate_evaluation_dir, load_pickle

MODEL_NAME_MAP = dict(
    bert_only_classification='classification',
    bert_only_embedding_classification='embedding,\nclassification',
    bert_all='all',
    bert_only_embedding='embedding',
)


def compute_average_score_per_repetition(data: pd.DataFrame) -> pd.DataFrame:
    results = list()
    for k, df_dataset_type in data.groupby(by='dataset_type'):
        for j, df_repetition in df_dataset_type.groupby(by='model_repetition_number'):
            for i, df_xai_method in df_repetition.groupby(by='attribution_method'):
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
    base_output_dir: str,
) -> None:
    data['mapped_model_name'] = data['model_name'].map(lambda x: MODEL_NAME_MAP[x])
    if 'top_k_precision' != metric:
        g = sns.catplot(
            data=data,
            x='mapped_model_name',
            # x='attribution_method',
            y=metric,
            hue='attribution_method',
            # hue='dataset_type',
            # split=True,
            # row='num_gaussians',
            col='dataset_type',
            # col='mapped_model_name',
            kind='violin',
            palette=sns.color_palette(palette='pastel'),
            fill=True,
            linewidth=0.0,
            height=2.5,
            inner_kws=dict(box_width=2, whis_width=0.2, color='0.4', marker='o'),
            # inner='stick',
            estimator='median',
            # errorbar=('pi', 95) if 'top_k_precision' != metric else 'sd',
            # errorbar='sd',
            # showfliers=False,
            # medianprops={'color': 'white', 'linewidth': 1.0},
            aspect=2.0,
            margin_titles=True,
            # line_kws={'linewidth': 1.5},
            facet_kws={'gridspec_kws': {'wspace': 0.1, 'hspace': 0.1}},
        )
        # g.map(
        #     sns.boxplot,
        #     'mapped_model_name',
        #     metric,
        #     'attribution_method',
        #     # estimator='median'
        #     notch=False,
        #     # whis=0,
        #     showfliers=False,
        #     showbox=False,
        #     showcaps=False,
        # )
    else:
        average_data = compute_average_score_per_repetition(data=data)
        g = sns.catplot(
            data=average_data,
            x='mapped_model_name',
            y=metric,
            hue='attribution_method',
            # row='num_gaussians',
            col='dataset_type',
            kind='bar',
            # linewidth=0.3,
            palette=sns.color_palette('pastel'),
            height=2.5,
            # inner='stick',
            estimator='mean',
            # errorbar=('pi', 95) if 'top_k_precision' != metric else 'sd',
            errorbar='sd',
            # errorbar=None,
            # errwidth=0.9,
            err_kws={'linewidth': 2.0},
            # showfliers=False,
            # medianprops={'color': 'black', 'linewidth': 1.0},
            aspect=1.0,
            margin_titles=True,
            # line_kws={'linewidth': 1.5},
            facet_kws={'gridspec_kws': {'wspace': 0.1, 'hspace': 0.1}},
        )

    for k in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):
            g.axes[k, j].grid(alpha=0.8, linewidth=0.5)
            # g.axes[k, j].title.set_size(6)
            # g.axes[k, j].set_xticklabels('')
            # g.axes[k, j].set_xlabel('')
            if 0 == k and 'top_k_precision' == metric:
                g.axes[k, j].set_ylabel(f'Average {metric}')
            g.axes[k, j].set_ylim(0, 1)
            g.axes[k, j].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            for label in (
                g.axes[k, j].get_xticklabels() + g.axes[k, j].get_yticklabels()
            ):
                label.set_fontsize(4)

    file_path = join(base_output_dir, f'{metric}.png')
    plt.savefig(file_path, dpi=300)
    plt.close()


def create_evaluation_plots(base_output_dir: str, config: dict) -> None:
    evaluation_dir = generate_evaluation_dir(config=config)
    file_path = join(evaluation_dir, config['evaluation']['evaluation_records'])
    evaluation_results = pd.DataFrame(load_pickle(file_path=file_path))
    visualization_methods = dict(
        roc_auc=plot_evaluation_results,
        precision_recall_auc=plot_evaluation_results,
        avg_precision=plot_evaluation_results,
        precision_specificity=plot_evaluation_results,
        top_k_precision=plot_evaluation_results,
    )

    plot_types = config['visualization']['base_data_type']['evaluation']
    for plot_type in plot_types:
        logger.info(f'Type of plot: {plot_type}')
        v = visualization_methods.get(plot_type, None)
        if v is not None:
            v(evaluation_results, plot_type, base_output_dir)


def visualize_results(base_output_dir: str, config: dict) -> None:
    for base_data_type, _ in config['visualization']['base_data_type'].items():
        v = VISUALIZATIONS.get(base_data_type, None)
        if v is not None:
            v(base_output_dir, config)


VISUALIZATIONS = dict(
    # data=create_data_plots,
    # xai=create_xai_plots,
    evaluation=create_evaluation_plots
    # model=create_model_performance_plots
)


def main(config: dict) -> None:
    output_dir = generate_visualization_dir(config=config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    visualize_results(base_output_dir=output_dir, config=config)


if __name__ == "__main__":
    main(config={})
