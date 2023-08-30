from collections import Counter
from copy import deepcopy
from os.path import join
from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from utils import generate_visualization_dir, generate_evaluation_dir, load_pickle
from IPython.display import HTML


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


def plot_most_common_xai_attributions(
    data: pd.DataFrame,
    plot_type: str,
    base_output_dir: str,
) -> None:
    data['mapped_model_name'] = data['model_name'].map(lambda x: MODEL_NAME_MAP[x])

    rows = np.unique(data['mapped_model_name'].values)
    columns = np.unique(data['dataset_type'].values)
    attribution_methods = np.unique(data['attribution_method'].values)
    ranks = np.unique(data['rank'].values)

    fig, axs = plt.subplots(
        nrows=len(rows),
        ncols=len(columns),
        sharex=True,
        sharey=True,
        layout='constrained',
        gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
        figsize=(4, 8),
    )

    grouped_data = data.groupby(by=['mapped_model_name', 'dataset_type'])

    for k, r in enumerate(rows):
        for j, c in enumerate(columns):
            for keys, df in grouped_data:
                if (r, c) != keys:
                    continue
                for normalized, gdf in df.groupby(by='normalized'):
                    if normalized:
                        continue
                    for s, (gender, ggdf) in enumerate(gdf.groupby(by='gender')):
                        ggdf[gender] = (
                            ggdf['attribution']
                            if 1 == s
                            else (-1) * ggdf['attribution']
                        )

                        g = sns.barplot(
                            data=ggdf,
                            x=gender,
                            y='rank',
                            order=ranks,
                            hue='attribution_method',
                            orient='y',
                            ax=axs[k, j],
                            width=0.8,
                            native_scale=False,
                            legend=True if 1 == s else False,
                            palette=sns.color_palette(
                                'pastel', len(attribution_methods)
                            )
                            if 1 == s
                            else sns.color_palette('muted', len(attribution_methods)),
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
                        axs[k, j].axvline(
                            x=0, color='black', linestyle='-', linewidth=0.5
                        )

                        axs[k, j].set_xticks(
                            [-1000, -500, 0, 500, 1000], [1000, 500, 0, 500, 1000]
                        )
                        axs[k, j].set_yticks(ranks, ranks + 1)
                        axs[k, j].set_xlabel(
                            'Cumulated attribution for female/male', fontsize=4
                        )
                        axs[k, j].set_ylabel(
                            ggdf.loc[ggdf.index[0], 'model_name'], fontsize=4
                        )
                        axs[k, j].spines['top'].set_linewidth(0.5)
                        axs[k, j].spines['right'].set_linewidth(0.5)
                        axs[k, j].spines['bottom'].set_linewidth(0.5)
                        axs[k, j].spines['left'].set_linewidth(0.5)
                        axs[k, j].grid(linewidth=0.2)
                        if 0 == k:
                            axs[k, j].set_title(
                                f'Dataset: {ggdf.loc[ggdf.index[0], "dataset_type"]}',
                                fontsize=4,
                            )

    file_path = join(base_output_dir, f'{plot_type}.png')
    logger.info(file_path)
    plt.savefig(file_path, dpi=300)
    plt.close()


def plot_attributions_per_sentence(
        data: np.ndarray,
        sentence: str
) -> None:
    """
    Source: https://captum.ai/tutorials/Image_and_Text_Classification_LIME
    """
    attrs = data
    rgb = lambda x: '255,0,0' if x < 0 else '0,255,0'
    alpha = lambda x: abs(x) ** 0.5
    token_marks = [
        f'<mark style="background-color:rgba({rgb(attr)},{alpha(attr)})">{token}</mark>'
        for token, attr in zip(sentence, attrs.tolist())
    ]    
    return HTML('<p>' + ' '.join(token_marks) + '</p>')


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


def create_dataset_for_xai_plot(plot_type: str, data: pd.DataFrame) -> pd.DataFrame:
    # d = '/home/rick/research/xai-nlp-benchmark/artifacts/nlp-benchmark-2023-08-23-15-26-05/visualization'
    # dump_as_pickle(data=data, output_dir=d, filename='xai_records_dataframe.pkl')
    output = None
    if 'most_common_xai_attributions' == plot_type:
        grouped_data = data.groupby(
            by=['model_name', 'dataset_type', 'target', 'attribution_method']
        )
        data_dict = dict(
            gender=list(),
            dataset_type=list(),
            model_name=list(),
            word=list(),
            attribution=list(),
            attribution_method=list(),
            rank=list(),
            normalized=list(),
        )

        for keys, df in tqdm(grouped_data):
            word_frequencies = dict()
            accumulated_attributions = dict()
            normalized_attributions = dict()
            for j, row in df.iterrows():
                for k, word in enumerate(row['sentence']):
                    if word not in accumulated_attributions:
                        accumulated_attributions[word] = row['attribution'][k]
                        word_frequencies[word] = 1
                    else:
                        accumulated_attributions[word] += row['attribution'][k]
                        word_frequencies[word] += 1

            for word in accumulated_attributions:
                r = accumulated_attributions[word] / word_frequencies[word]
                normalized_attributions[word] = r

            word_counter = Counter(accumulated_attributions)
            word_counter_normalized = Counter(normalized_attributions)
            word_counters = zip(
                word_counter.most_common(n=5),
                word_counter_normalized.most_common(n=5),
            )

            for i, (c, cn) in enumerate(word_counters):
                data_dict['model_name'] += [keys[0]]
                data_dict['dataset_type'] += [keys[1]]
                data_dict['gender'] += [GENDER[keys[2]]]
                data_dict['attribution_method'] += [keys[3]]
                data_dict['word'] += [c[0]]
                data_dict['attribution'] += [c[1]]
                data_dict['rank'] += [i]
                data_dict['normalized'] += [0]

                data_dict['model_name'] += [keys[0]]
                data_dict['dataset_type'] += [keys[1]]
                data_dict['gender'] += [GENDER[keys[2]]]
                data_dict['attribution_method'] += [keys[3]]
                data_dict['word'] += [cn[0]]
                data_dict['attribution'] += [cn[1]]
                data_dict['rank'] += [i]
                data_dict['normalized'] += [1]

        output = pd.DataFrame(data_dict)
    return output

    # d = '/home/rick/research/xai-nlp-benchmark/artifacts/nlp-benchmark-2023-08-23-15-26-05/visualization'
    # dump_as_pickle(data=results, output_dir=d, filename='results.pkl')
    # results = load_pickle(file_path=join(d, 'results.pkl'))


def load_xai_records(config: dict) -> pd.DataFrame:
    xai_dir = generate_xai_dir(config=config)
    file_path = join(xai_dir, config['xai']['xai_records'])
    paths_to_xai_records = load_pickle(file_path=file_path)
    data_list = list()
    for p in tqdm(paths_to_xai_records):
        local_path = join(*p.split('/')[2:])
        results = load_pickle(file_path=local_path)
        for xai_records in results:
            data_list += [asdict(xai_records)]

    return pd.DataFrame(data_list)


def create_xai_plots(base_output_dir: str, config: dict) -> None:
    xai_records = load_xai_records(config=config)
    visualization_methods = dict(
        most_common_xai_attributions=plot_most_common_xai_attributions,
    )

    plot_types = config['visualization']['visualizations']['xai']
    for plot_type in plot_types:
        logger.info(f'Type of plot: {plot_type}')
        v = visualization_methods.get(plot_type, None)
        if v is None:
            continue
        data = create_dataset_for_xai_plot(plot_type=plot_type, data=xai_records)
        v(data, plot_type, base_output_dir)


def visualize_results(base_output_dir: str, config: dict) -> None:
    for visualization, _ in config['visualization']['visualizations'].items():
        v = VISUALIZATIONS.get(visualization, None)
        if v is not None:
            v(base_output_dir, config)


VISUALIZATIONS = dict(
    # data=create_data_plots,
    xai=create_xai_plots,
    evaluation=create_evaluation_plots,
    model=create_model_performance_plots,
)


def main(config: dict) -> None:
    output_dir = generate_visualization_dir(config=config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    visualize_results(base_output_dir=output_dir, config=config)


if __name__ == "__main__":
    main(config={})
