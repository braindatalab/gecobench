import os
from os.path import join
from os import listdir
from os.path import isfile, join
from pathlib import Path
from itertools import islice

import pandas as pd
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import ast

from utils import generate_visualization_dir, generate_evaluation_dir, load_pickle, dump_as_pickle
from dataclasses import asdict

path_to_xai_records_file = '/Users/arturdox/coding/qailabs/xai-nlp-benchmark/artifacts/nlp-benchmark-2023-08-23-15-26-05-V2/xai/xai_records.pkl'

def load_data_xai_visualization(path_to_xai_records_file: list):
    xai_results_paths = load_pickle(file_path=path_to_xai_records_file)
    # remove /mnt in and adjust path
    xai_results_paths = [w.replace('/mnt/', '') for w in xai_results_paths]
    xai_results_paths = [w.replace('nlp-benchmark-2023-08-23-15-26-05', 'nlp-benchmark-2023-08-23-15-26-05-V2') for w in xai_results_paths]
    data = list()
    for path in xai_results_paths[:10]: 
        list_of_xai_results = load_pickle(file_path=path)
        for result in list_of_xai_results:
            data_dict = asdict(result)
            data_dict['sentence'] = str(data_dict['sentence'])
            data += [data_dict]
    xai_results = pd.DataFrame(data)
    xai_results_grouped = xai_results.groupby(["model_name", "dataset_type", "sentence"])
    return xai_results_grouped


def plot_attributions(
        xai_results_grouped: pd.DataFrame
) -> None:

    # islice for selecting a specific sample
    for i, dataframe in islice(xai_results_grouped, 4, None):
        sentence = ast.literal_eval(dataframe['sentence'].iloc[0])
        xai_methods_per_sentence = dataframe['attribution_method']
        attribution_scores_per_sentence = dataframe['attribution']
        ground_truth_per_sentence = dataframe['ground_truth']

        # sanity check
        assert len(sentence) == len(attribution_scores_per_sentence[:6].iloc[0])
   
        # pick one model (model repition number)
        attribution_scores_per_sentence = attribution_scores_per_sentence[:6]
        xai_methods_per_sentence = xai_methods_per_sentence[:6]
        ground_truth_per_sentence = ground_truth_per_sentence[:6].iloc[0]

        sentences_w_ground_truths = list(zip(sentence, ground_truth_per_sentence))
        word_idx = 0

        image_paths = []
        for word in range(len(sentence)):
            attribution_scores_per_word = []
            xai_methods_per_word = [] 

            for method in range(len(attribution_scores_per_sentence)):
                attribution_scores_per_word.append(attribution_scores_per_sentence.iloc[method][word])
                xai_methods_per_word.append(xai_methods_per_sentence.iloc[method])
            
            g = sns.barplot(
                x=xai_methods_per_word, 
                y=attribution_scores_per_word,
            )
            
            # GPT4-generated code
            g.set_xticklabels([])
            for bar, label in zip(g.patches, xai_methods_per_word):
                height = bar.get_height()  
                g.text(
                    bar.get_x() + bar.get_width() / 2,  # X position is the center of the bar
                    height + 0.04,                      # Y position is at the top of the bar
                    label,                              # The text to display
                    ha='center',                        # Center the text horizontally
                    va='bottom',                        # Position the text above the bar
                    rotation=90
                )

            plt.yticks(np.arange(0, 1.1, 0.1))
            dir_path = "/Users/arturdox/Downloads/xai_sentence/"
            save_path = dir_path + str(word_idx) + "_attributions_word_"+sentence[word]+".png"
            image_paths.append(save_path)
            plt.savefig(save_path , dpi=300)
            plt.close()

            word_idx += 1

        # GPT4-generated code
        html_content = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Concatenated Images</title>
            <style>
                .image-container{
                    display: inline-flex;
                    flex-direction: row;
                    justify-content: flex-start;
                    align-items: center;
                }
                .image-box {
                    margin-right: 20px; /* Adjust spacing between image-text blocks */
                }
                .image-box img {
                    max-width: 120px; /* Set a maximum width for each image */
                    max-height: 120px; /* Set a maximum height for each image */
                    object-fit: contain; /* Ensure the aspect ratio of images is maintained */
                    display: block; /* Makes the image a block-level element */
                    margin-bottom: 5px; /* Spacing between image and text */
                }
                .image-text {
                    text-align: center; /* Center-aligns the text below the image */
                }
                .highlight {
                    background-color: lightgrey; /* Highlight color */
                    border-radius: 5px;
                }
            </style>
        </head>
        <body>
            <div class="image-container">
        '''

        for img_path, (text, highlight) in zip(image_paths, sentences_w_ground_truths):
            if os.path.exists(img_path):
                highlight_class = 'highlight' if highlight else ''
                html_content += f'''
                <div class="image-box">
                    <img src="{img_path}" alt="Image">
                    <div class="image-text {highlight_class}">{text}</div>
                </div>
                '''
            else:
                print(f"Warning: Image {img_path} not found.")

        html_content += '''
            </div> <!-- Closing image-container -->
        </body>
        </html>
        '''

        html_file = '/Users/arturdox/Downloads/V3_plot_gt_highlited_FINAL.html'
        with open(html_file, 'w') as file:
            file.write(html_content)

        break
        
    return None


def plot_evaluation_results(
        data: pd.DataFrame,
        metric: str,
        base_output_dir: str,
) -> None:
        g = sns.catplot(
            data=data, x='model_name', y=metric,
            hue='attribution_method',
            # row='num_gaussians',
            col='dataset_type', kind='box', linewidth=0.3,
            height=2.5,
            # inner='stick',
            estimator='median',
            # errorbar=('pi':, 75),
            # errorbar='sd',
            showfliers=False,
            medianprops={'color': 'black', 'linewidth': 1.0},
            aspect=1., margin_titles=True,
            # line_kws={'linewidth': 1.5},
            facet_kws={'gridspec_kws': {'wspace': 0.1, 'hspace': 0.1}},
        )
        file_path = join(base_output_dir, f'{metric}.png')
        plt.savefig(file_path, dpi=300)
        plt.close()


def visualize_results(base_output_dir: str, config: dict) -> None:
    for result_type, _ in config['visualization']['visualizations'].items():
        v = VISUALIZATIONS.get(result_type, None)
        if v is not None:
            v(base_output_dir, config)


def create_evaluation_plots(base_output_dir: str, config: dict) -> None:
    evaluation_dir = generate_evaluation_dir(config=config)
    file_path = join(evaluation_dir, config['evaluation']['evaluation_records'])
    evaluation_results = pd.DataFrame(load_pickle(file_path=file_path))
    visualization_methods = dict(
        roc_auc=plot_evaluation_results,
        # precision_recall_auc=plot_evaluation_results,
        # avg_precision=plot_evaluation_results,
        # precision_specificity=plot_evaluation_results,
        # top_k_precision=plot_evaluation_results
    )

    plot_types = config['visualization']['visualizations']['evaluation']
    for plot_type in plot_types:
        logger.info(f'Type of plot: {plot_type}')
        v = visualization_methods.get(plot_type, None)
        if v is not None:
            v(evaluation_results, plot_type, base_output_dir)


VISUALIZATIONS = dict(
    # data=create_data_plots,
    # xai=create_xai_plots,
    # evaluation=create_evaluation_plots
    # model=create_model_performance_plots
)


def main(config: dict) -> None:
    # use output_dir for visualization
    output_dir = generate_visualization_dir(config=config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    visualize_results(base_output_dir=output_dir, config=config)

    #xai_records_file = create_xai_records_list()
    xai_results_grouped = load_data_xai_visualization(path_to_xai_records_file)
    plot_attributions(xai_results_grouped)


if __name__ == "__main__":
    main(config={})
