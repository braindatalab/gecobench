import pickle
from typing import Dict

import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients
from IPython.display import display, HTML
from transformers import BertTokenizer

from training.main import dataset_pre_processing

DEVICE = 'cpu'


def custom_forward_grad(inputs, model):
    preds = model(inputs)[0]
    y = torch.softmax(preds, dim=1)[0][1].unsqueeze(-1)
    return y


def colorize(words, gt, attrs):
    # gt=gt.drop(['index'],axis=1)
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words

    # CLS
    color = _get_color(summarize_attributions(attrs)[0].item())
    colored_string = f'<span style="color: black; background-color: {color}">{words[0]}</span>'

    # sentence words (which are the ones that exist in the ground truth)
    for i, word in enumerate(words[1:-1], start=1):
        color = _get_color(summarize_attributions(attrs)[i].item())

        #         if int(gt[i-1].item())==1:
        #             template = f'<span  style="color:black; border: 3px solid black;  background-color: {color}"> {word}</span> '
        #         else:
        #             template = f'<span style="color: black; background-color: {color}">{word}</span> '
        template = f'<span style="color: black; background-color: {color}">{word}</span> '

        colored_string += template

    # SEP
    color = _get_color(summarize_attributions(attrs)[-1].item())
    colored_string += f'<span style="color: black; background-color: {color}">{words[-1]}</span>'

    return colored_string


def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    if attr > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    colour = f"hsl({hue}, {sat}%, {lig}%)"
    return colour


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def highlight_words(gt, model, tokenizer, validation_set, n, show_highlight=True):
    # encode text: from a dataset of phrase and target, we locate the text, target and associate a ground-truth.
    # Then the text is tokenized.
    validation_set = pd.DataFrame(validation_set, columns=['phrases', 'target'])

    text = validation_set['phrases'].loc[n]
    print(text)
    # gt_ph = gt.iloc[n]
    true_label = validation_set['target'].loc[n]
    tokenized = tokenizer.encode(text)

    input_ids = torch.tensor([tokenized]).to(DEVICE)
    score = model(input_ids)[0]
    score_list = [round(x, 4) for x in score.detach().tolist()[0]]
    pred_label = score_list.index(max(score_list))

    ref_token_id = tokenizer.pad_token_id  # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id  # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id  # A token used for prepending to the concatenated question-text word sequence

    ref_input_ids = [tokenizer.cls_token_id] + [tokenizer.pad_token_id] * (
            len([i for i in tokenized if i != 0]) - 2) + [tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (
                        len([i for i in tokenized if i == 0]))
    ref_input_ids = torch.tensor([ref_input_ids])

    lig = LayerIntegratedGradients(custom_forward_grad, model.bert.embeddings)

    attributions, delta = lig.attribute(inputs=input_ids.long(),
                                        baselines=ref_input_ids.to(DEVICE),
                                        #                                     internal_batch_size=3,
                                        return_convergence_delta=True)

    words = [tokenizer.decode(t).replace(' ', '') for t in tokenized]
    s = colorize(words, None, attributions)

    if show_highlight:
        print('correct label:', true_label, 'predicted label', pred_label)
        display(HTML(s))

    return int(true_label), pred_label


def main(config: Dict) -> None:
    with open('../data/df_validation_male_all.pkl', 'rb') as f:
        val_all_male = pickle.load(f)

    with open('../data/df_validation_female_all.pkl', 'rb') as f:
        val_all_female = pickle.load(f)

    tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')

    _, target_all_f, _, _, encoded_train_phrases_all_f, _ = dataset_pre_processing(
        val_all_male, tokenizer1)
    _, target_all_m, _, _, encoded_train_phrases_all_f, _ = dataset_pre_processing(
        val_all_female, tokenizer1)

    model = torch.load('../artifacts/bert_model.pt', map_location=DEVICE)

    highlight_words(
        gt=None,
        model=model,
        tokenizer=tokenizer1,
        validation_set=val_all_male,
        n=2,
        show_highlight=True)


if __name__ == '__main__':
    main(config={})
