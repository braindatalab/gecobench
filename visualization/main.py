import ast
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import islice
from os.path import join, exists
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from pandas.core.groupby.generic import DataFrameGroupBy
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from common import DatasetKeys

from utils import (
    generate_visualization_dir,
    generate_evaluation_dir,
    load_pickle,
    load_jsonl_as_df,
    generate_training_dir,
    generate_xai_dir,
    generate_project_dir,
    generate_data_dir
)

MODEL_NAME_MAP = dict(
    bert_only_classification='newly initialized\nclassification',
    bert_only_embedding_classification='fine-tuned embedding,\nnewly initialized classification',
    bert_all='fine-tuned all',
    bert_only_embedding='fine-tuned embedding',
    bert_randomly_init_embedding_classification='newly initialized embedding,\nclassification',
)

DATASET_NAME_MAP = dict(subject='$\mathcal{D}_{S}$', all='$\mathcal{D}_{SO}$')
GENDER = {0.0: 'female', 1.0: 'male'}


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
                            palette=(
                                sns.color_palette('pastel', len(attribution_methods))
                                if 1 == s
                                else sns.color_palette(
                                    'muted', len(attribution_methods)
                                )
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
                            ggdf.loc[ggdf.index[0], 'mapped_model_name'], fontsize=4
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
        mass_accuracy=plot_evaluation_results,
    )

    plot_types = config['visualization']['visualizations']['evaluation']
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
            # history_path = join(*record[-1].split('/')[2:])
            history_path = record[-1]
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

    plot_types = config['visualization']['visualizations']['model']
    for plot_type in plot_types:
        logger.info(f'Type of plot: {plot_type}')
        v = visualization_methods.get(plot_type, None)
        if v is None:
            continue
        v(history, plot_type, base_output_dir)


def create_xai_sentence_html_plots(
    data: DataFrameGroupBy, plot_type: str, base_output_dir: str
) -> None:
    # Using islice for selecting a specific sentence
    sample_index = 1179
    for i, dataframe in islice(data, sample_index, None):
        sentence = ast.literal_eval(dataframe['sentence'].iloc[0])
        selected_sentence_length = len(sentence)
        break
    dataset_type = i[1]

    # Search for corresponding samples with above properties but from different pre-trained models
    pre_trained_models = [
        'bert_all',
        'bert_only_embedding_classification',
        'bert_only_classification',
        'bert_only_embedding',
    ]
    df_explanations_sentence_different_models = pd.DataFrame()
    for key, group in data:
        for index, row in group.iterrows():
            for model in pre_trained_models:
                if row['model_name'] == model and row['model_repetition_number'] == 0 and row['sentence'] == str(sentence) and row['dataset_type'] == str(dataset_type):
                    df_explanations_sentence_different_models = df_explanations_sentence_different_models.append(row, ignore_index=True)
                    
    model_image_paths = []
    for model in pre_trained_models:
        df_model = df_explanations_sentence_different_models[
            df_explanations_sentence_different_models['model_name'] == model
        ]
        sentence = ast.literal_eval(df_model['sentence'].iloc[0])
        xai_methods_per_sentence = df_model['attribution_method']
        attribution_scores_per_sentence = df_model['attribution']
        ground_truth_per_sentence = df_model['ground_truth'].iloc[0]

        sentences_w_ground_truths = list(zip(sentence, ground_truth_per_sentence))

        word_idx = 0
        image_paths = []
        for word in range(len(sentence)):
            attribution_scores_per_word = []
            xai_methods_per_word = []

            for method in range(len(attribution_scores_per_sentence)):
                attribution_scores_per_word.append(
                    attribution_scores_per_sentence.iloc[method][word]
                )
                xai_methods_per_word.append(xai_methods_per_sentence.iloc[method])

            # Save only plot legend as separte figure to be appended in HTML file at the bottom
            folder_path = join(base_output_dir, f"{model}_xai_attributions_per_word")
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            file_path_legend_plot = join(folder_path, f'plot_legend.png')

            h = sns.barplot(
                x=xai_methods_per_word,
                y=attribution_scores_per_word,
                hue=xai_methods_per_word,
                # width=0.8
            )

            # GPT4-generated code to create legend of barplot and save it as a figure used in the final plot
            # Previous approach: handles, labels = h.get_legend_handles_labels() with
            # ax_legend.legend(handles, labels,...) stopped working

            # Get the unique colors of the bars
            colors = [p.get_facecolor() for p in h.patches]
            # Create custom legend
            legend_patches = [
                plt.Rectangle((0, 0), 1, 1, facecolor=colors[i])
                for i in range(len(xai_methods_per_word))
            ]

            # Create a new figure for the legend
            fig_legend = plt.figure(figsize=(3, 3))
            ax_legend = fig_legend.add_subplot(111)
            ax_legend.legend(
                handles=legend_patches,
                labels=xai_methods_per_word,
                loc='center',
                ncol=len(legend_patches),
                frameon=False,
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

            # Set the size of the figure (width, height)
            plt.figure(figsize=(3, 2))

            # Barplots for each word
            g = sns.barplot(
                x=xai_methods_per_word,
                y=attribution_scores_per_word,
                hue=xai_methods_per_word,
                width=0.8,
            )

            sns.despine(left=True, bottom=True)
            g.set_yticklabels([])
            g.tick_params(left=False)

            # GPT4-generated code
            g.set_xticklabels([])
            # for bar, label in zip(g.patches, xai_methods_per_word):
            #     height = bar.get_height()
            #     g.text(
            #         bar.get_x()
            #         + bar.get_width() / 2,  # X position is the center of the bar
            #         height + 0.04,  # Y position is at the top of the bar
            #         label,  # The text to display (name of XAI method)
            #         ha='center',  # Center the text horizontally
            #         va='bottom',  # Position the text above the bar
            #         rotation=90,
            #     )

            plt.yticks(np.arange(0, 1.1, 0.1))
            folder_path = join(base_output_dir, f"{model}_xai_attributions_per_word")
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            file_path = join(
                folder_path, f'{str(word_idx)}_attributions_word_{sentence[word]}.png'
            )
            image_paths.append(file_path)

            plt.tight_layout()
            plt.savefig(file_path, dpi=300)
            plt.close()

            word_idx += 1
        model_image_paths.append(image_paths)

    # GPT4-generated code
    html_content = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Concatenated Images</title>
        <style>
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
                margin-right: -5px; /* Adjust spacing between image-text blocks */
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
                border-radius: 0px;
            }
            .legend-plot img {
                max-width: 800px; /* Adjust max width as needed */
                max-height: 800px; /* Adjust max height for vertical image */
                object-fit: contain;
            }
            .image-with-caption {
                display: flex;
                align-items: center; /* Vertically center the flex items */
            }
            .text-container {
                width: 50px; /* Fixed width for the text container */
                text-align: right; /* Align text to the right */
                margin-right: 5px; /* Consistent margin to the right of the text */
            }
        </style>
    </head>
    <body>
        <div class="image-container">
    '''

    model01 = ["" for _ in range(len(sentences_w_ground_truths))]
    model01[0] = "All "
    model02 = ["" for _ in range(len(sentences_w_ground_truths))]
    model02[0] = "EmdC"
    model03 = ["" for _ in range(len(sentences_w_ground_truths))]
    model03[0] = "C   "
    model04 = ["" for _ in range(len(sentences_w_ground_truths))]
    model04[0] = "Emd "

    image_model_captions_zipped = zip(model01, model02, model03, model04)
    image_model_captions_zipped = [list(group) for group in image_model_captions_zipped]

    model_image_paths_zipped = [list(group) for group in zip(*model_image_paths)]
    for index, (model_name_caption, img_path, (text, highlight)) in enumerate(
        zip(
            image_model_captions_zipped,
            model_image_paths_zipped,
            sentences_w_ground_truths,
        )
    ):
        if exists(img_path[0]) and exists(
            img_path[1] and exists(img_path[2]) and exists(img_path[3])
        ):
            if index == 0:
                highlight_class = 'highlight' if highlight else ''
                html_content += f'''
                <div class="image-box">
                    <div class="image-with-caption">
                        <div class="text-container">{model_name_caption[0]}</div>
                            <div class="image-model-one">
                                <img src="{img_path[0]}" alt="Image">
                            <div class="image-text {highlight_class}">{text}</div>
                        </div>
                    </div>
                    
                    <div class="image-with-caption">
                        <div class="text-container">{model_name_caption[1]}</div>
                            <div class="image-model-two">
                                <img src="{img_path[1]}" alt="Image">
                            <div class="image-text {highlight_class}">{text}</div>
                        </div>
                    </div>
                                    
                    <div class="image-with-caption">
                        <div class="text-container">{model_name_caption[2]}</div>
                            <div class="image-model-two">
                                <img src="{img_path[2]}" alt="Image">
                            <div class="image-text {highlight_class}">{text}</div>
                        </div>
                    </div>

                    <div class="image-with-caption">
                        <div class="text-container">{model_name_caption[3]}</div>
                            <div class="image-model-two">
                                <img src="{img_path[3]}" alt="Image">
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
                        <img src="{img_path[0]}" alt="Image">
                    </div>
                    <div class="image-text {highlight_class}">{text}</div>

                    <div class="image-model-two">
                        <img src="{img_path[1]}" alt="Image">
                    </div>
                    <div class="image-text {highlight_class}">{text}</div>
                                    
                    <div class="image-model-three">
                        <img src="{img_path[2]}" alt="Image">
                    </div>
                    <div class="image-text {highlight_class}">{text}</div>
                                    
                    <div class="image-model-fourth">
                        <img src="{img_path[3]}" alt="Image">
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
                <img src="{file_path_legend_plot}" alt="Legend Plot">
            </div>
        </body>
        </html>
        '''
    else:
        print(f"Warning: Image {file_path_legend_plot} not found.")

    file_path = join(base_output_dir, f'{sample_index}_{dataset_type}_{selected_sentence_length}_models_xai_sentence_html_plot.html')
    with open(file_path, 'w') as file:
        file.write(html_content)


def create_dataset_for_xai_plot(
    plot_type: str, xai_records: list
) -> pd.DataFrame | DataFrameGroupBy:
    output = None
    if 'most_common_xai_attributions' == plot_type:
        data = pd.DataFrame(xai_records)
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
    elif 'sentence_html_plot' == plot_type:
        data_list = list()
        for record in xai_records:
            new_record = deepcopy(record)
            new_record['sentence'] = str(record['sentence'])
            data_list.append(new_record)
        output = pd.DataFrame(data_list).groupby(
            ["model_name", "dataset_type", "sentence"]
        )
    return output


def load_xai_records(config: dict) -> list:
    xai_dir = generate_xai_dir(config=config)
    file_path = join(xai_dir, config['xai']['xai_records'])
    paths_to_xai_records = load_pickle(file_path=file_path)
    # Temporary fix for running locally
    paths_to_xai_records = [s.strip('/mnt/') for s in paths_to_xai_records]
    data_list = list()
    for p in tqdm(paths_to_xai_records):
        results = load_pickle(file_path=p)
        for xai_records in results:
            data_list += [asdict(xai_records)]

    return data_list


def get_correctly_classified_records(records: list) -> list:
    result = list()
    # TODO: Where is the correct_classified_intersection field coming from?
    for r in records:
        if 0 == r['correct_classified_intersection']:
            continue
        result.append(deepcopy(r))
    return result


def create_xai_plots(base_output_dir: str, config: dict) -> None:
    xai_records = load_xai_records(config=config)
    #filtered_xai_records = get_correctly_classified_records(records=xai_records)
    visualization_methods = dict(
        most_common_xai_attributions=plot_most_common_xai_attributions,
        sentence_html_plot=create_xai_sentence_html_plots,
    )

    plot_types = config['visualization']['visualizations']['xai']
    for plot_type in plot_types:
        logger.info(f'Type of plot: {plot_type}')
        v = visualization_methods.get(plot_type, None)
        base_output_dir = (
            join(generate_project_dir(), base_output_dir)
            if plot_type == 'sentence_html_plot'
            else base_output_dir
        )
        if v is None:
            continue
        if 'sentence_html_plot' == plot_type:
            base_output_dir = join(generate_project_dir(), base_output_dir)
        data = create_dataset_for_xai_plot(
            plot_type=plot_type, xai_records=xai_records
        )
        v(data, plot_type, base_output_dir)


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
        # layout='constrained',
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
                g.bar_label(container, labels=[word], fontsize=3)
            axs[k].legend(
                loc='lower right',
                ncols=1,
                # fontsize='xx-small',
                prop={'size': 2},
            )

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

    data = dict(all=dataset_all, subject=dataset_subject)
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
)


def main(config: dict) -> None:
    output_dir = generate_visualization_dir(config=config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    visualize_results(base_output_dir=output_dir, config=config)


if __name__ == "__main__":
    main(config={})
