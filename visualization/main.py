from copy import deepcopy
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt

from utils import (
    generate_visualization_dir,
    generate_evaluation_dir,
    load_pickle,
    generate_training_dir,
)

MODEL_NAME_MAP = dict(
    bert_only_classification='classification',
    bert_only_embedding_classification='embedding,\nclassification',
    bert_all='all',
    bert_only_embedding='embedding',
)


def compute_average_score_per_repetition(data: pd.DataFrame) -> pd.DataFrame:
    results = list()
    for k, df_dataset_type in data.groupby(by='dataset_type'):
        for l, df_model_type in df_dataset_type.groupby(by='model_name'):
            for j, df_repetition in df_model_type.groupby(by='model_repetition_number'):
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
    def _plot_postprocessing(g):
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

    data['mapped_model_name'] = data['model_name'].map(lambda x: MODEL_NAME_MAP[x])
    if 'top_k_precision' != metric:
        average_data = compute_average_score_per_repetition(data=data)
        datasets = [('', data), ('averaged', average_data)]
        for s, d in datasets:
            g = sns.catplot(
                data=d,
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

            _plot_postprocessing(g=g)
            file_path = join(base_output_dir, f'{metric}_{s}.png')
            plt.savefig(file_path, dpi=300)
            plt.close()

    else:
        average_data = compute_average_score_per_repetition(data=data)
        datasets = [('', data), ('averaged', average_data)]
        for s, d in datasets:
            g = sns.catplot(
                data=d,
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

            _plot_postprocessing(g=g)
            file_path = join(base_output_dir, f'{metric}_{s}.png')
            plt.savefig(file_path, dpi=300)
            plt.close()


def plot_model_performance(
    data: pd.DataFrame,
    plot_type: str,
    base_output_dir: str,
) -> None:
    data['mapped_model_name'] = data['model_name'].map(lambda x: MODEL_NAME_MAP[x])
    g = sns.catplot(
        data=data,
        x='mapped_model_name',
        # x='attribution_method',
        y='accuracy',
        hue='data_split',
        # hue='dataset_type',
        # split=True,
        # row='num_gaussians',
        col='dataset_type',
        # col='mapped_model_name',
        kind='bar',
        palette=sns.color_palette(palette='pastel'),
        fill=True,
        linewidth=0.0,
        height=2.5,
        # inner_kws=dict(box_width=2, whis_width=0.2, color='0.4', marker='o'),
        # inner='stick',
        estimator='mean',
        # errorbar=('pi', 95) if 'top_k_precision' != metric else 'sd',
        errorbar='sd',
        # showfliers=False,
        # medianprops={'color': 'white', 'linewidth': 1.0},
        aspect=2.0,
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

            g.axes[k, j].set_ylim(0, 1)
            # g.axes[k, j].set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            g.axes[k, j].set_yticks(np.arange(start=0.1, step=0.1, stop=1.1))
            for label in (
                g.axes[k, j].get_xticklabels() + g.axes[k, j].get_yticklabels()
            ):
                label.set_fontsize(4)

    file_path = join(base_output_dir, f'{plot_type}.png')
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


def create_model_performance_plots(base_output_dir: str, config: dict) -> None:
    data_dict = dict(
        dataset_type=list(), model_name=list(), accuracy=list(), data_split=list()
    )

    def load_training_history(records: list) -> pd.DataFrame:
        for record in records:
            history_path = join(*record[-1].split('/')[2:])
            training_history = load_pickle(file_path=history_path)
            data_dict['dataset_type'] += [record[0].split('_')[-1]]
            data_dict['model_name'] += [record[1]['model_name']]
            data_dict['accuracy'] += [training_history['train_acc'][-1]]
            data_dict['data_split'] += ['training']
            data_dict['dataset_type'] += [record[0].split('_')[-1]]
            data_dict['model_name'] += [record[1]['model_name']]
            data_dict['accuracy'] += [training_history['val_acc'][-1]]
            data_dict['data_split'] += ['validation']
        return pd.DataFrame(data_dict)

    training_dir = generate_training_dir(config=config)
    file_path = join(training_dir, config['training']['training_records'])
    training_records = load_pickle(file_path=file_path)
    history = load_training_history(records=training_records)
    visualization_methods = dict(
        model_performance=plot_model_performance,
    )

    plot_types = config['visualization']['base_data_type']['model']
    for plot_type in plot_types:
        logger.info(f'Type of plot: {plot_type}')
        v = visualization_methods.get(plot_type, None)
        if v is None:
            continue
        v(history, plot_type, base_output_dir)


def visualize_results(base_output_dir: str, config: dict) -> None:
    for base_data_type, _ in config['visualization']['base_data_type'].items():
        v = VISUALIZATIONS.get(base_data_type, None)
        if v is not None:
            v(base_output_dir, config)


VISUALIZATIONS = dict(
    # data=create_data_plots,
    # xai=create_xai_plots,
    evaluation=create_evaluation_plots,
    model=create_model_performance_plots,
)


def main(config: dict) -> None:
    output_dir = generate_visualization_dir(config=config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    visualize_results(base_output_dir=output_dir, config=config)


if __name__ == "__main__":
    main(config={})
