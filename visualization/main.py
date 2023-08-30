from os.path import join
from pathlib import Path

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



def visualize_results(base_output_dir: str, config: dict) -> None:
    for result_type, _ in config['visualization']['visualizations'].items():
        v = VISUALIZATIONS.get(result_type, None)
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
