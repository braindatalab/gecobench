from os.path import join
from pathlib import Path

import pandas as pd
from loguru import logger
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from utils import generate_visualization_dir, generate_evaluation_dir, load_pickle
from IPython.display import HTML


"""
    Call 04.10.:
    # 0. Alle Daten Ordner vom Cluster holen und in neuen artifacts folder packen ()"Copy alle aus Dicetory -r")
    # 0. Generien einer Liste aller Pfade zu den Intermediate XAI Results Dateein 
    # 1. Group by Satz und Model Name und Model Reptition Number -> Dict mit 5 XAI Methoden
    # 2. Pro Wort die Attribution Scores -> Für Rechteck Visualisierung
    # 3. ... Ground Truth -> Für Rechteck Visualisierung
    
    # Logik: Auch über Modelle iterieren (aktuell nur 1 Modell)
    # Hinweis: Falls mehr attribution scores als Wörter - Rest abschneiden

"""

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


# ToDo: Not hardcoded
xai_records_paths = "/Users/arturdox/coding/qailabs/xai-nlp-benchmark/artifacts/nlp-benchmark-2023-05-23-11-48-27/xai/xai_records_test.pkl"
# intermediate_raw_xai_result-2023-09-20-16-05-48.pkl
# xai_records.pkl

def load_data_sentence_visualization(xai_records_paths: list):
    # 'path/to/xai_records.pkl'
    print("load_data_sentence_visualization")
    print("xai_records_paths:", xai_records_paths)
    xai_results_paths = load_pickle(file_path=xai_records_paths)
    print("xai_results_paths")
    print(xai_results_paths)

    data = list()
    for path in xai_results_paths:
        data += [load_pickle(file_path=path)]
    xai_results = pd.DataFrame(data)

    # get number of XAI methods
    methods = []
    for i in range(xai_results.shape[1]):
        methods.append(xai_results.iloc[0][i].attribution_method)
    num_xai_methods = len(set(methods))
    print(set(methods))

    print(xai_results.shape)
    print("#################")
    print(xai_results.iloc[0])
    print("#################")
    print("xai_results.iloc[0][0].model_repetition_number")
    print(xai_results.iloc[0][0].model_repetition_number)
    print(xai_results.iloc[0][0])
    print(xai_results.iloc[0][1])
    print(xai_results.iloc[0][2])
    print(xai_results.iloc[0][3])
    print(xai_results.iloc[0][4])
    print(xai_results.iloc[0][5])
    
    # hardcode one plot example
    sentence = xai_results.iloc[0][0+num_xai_methods].sentence
    attributions = []
    xai_methods = []
    for i in range(num_xai_methods):
        print(i+num_xai_methods)
        print(xai_results.iloc[0][i+num_xai_methods].sentence)
        print(xai_results.iloc[0][i+num_xai_methods].raw_attribution)
        attributions.append(xai_results.iloc[0][i+num_xai_methods].raw_attribution)
        xai_methods.append(xai_results.iloc[0][i+num_xai_methods].attribution_method)
    print(sentence)
    attributions = np.array(attributions)
    print(np.array(attributions))

    html_file = plot_attributions_per_sentence(attributions, sentence, xai_methods)
    with open('/Users/arturdox/Downloads/one_sentence_2.html', 'w') as f:
        f.write(html_file.data)

    print("#########################")
    print(xai_results.iloc[0][0+num_xai_methods].sentence)

def plot_attributions_per_sentence(
        attrs: np.ndarray,
        sentence: str,
        xai_methods: list
) -> None:
    """
    Adapted from https://captum.ai/tutorials/Image_and_Text_Classification_LIME
    """
    rgb = lambda x: '255,0,0' if x < 0 else '0,255,0'
    alpha = lambda x: abs(x) ** 0.5
    
    html_string = ""

    #print("plot_attributions_per_sentence")
    #print(sentence)
    #print(xai_methods)
    print(attrs)

    for idx, xai_attribution in enumerate(attrs):
        token_marks = [
            f'<mark style="background-color:rgba({rgb(attr)},{alpha(attr)})">{token}</mark>'
            for token, attr in zip(sentence, xai_attribution.tolist())
            ]
        #print(xai_methods[idx])
        html_string = html_string + 'XAI Method: ' + xai_methods[idx] + '<p>' + ' '.join(token_marks) + '</p>' + '</br>'
     
    return HTML(html_string)


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
    # use output_dir for visualization
    output_dir = generate_visualization_dir(config=config)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    visualize_results(base_output_dir=output_dir, config=config)
    load_data_sentence_visualization(xai_records_paths)


if __name__ == "__main__":
    main(config={})
