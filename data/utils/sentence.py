import pandas as pd

from utils import dump_as_jsonl


def dump_df_as_jsonl(df: pd.DataFrame, output_dir: str, filename: str) -> None:
    word_list = df['text'].apply(lambda x: x.split(" ")).tolist()
    target = df['target'].tolist()

    set = [
        {"sentence": word_list_item, "target": target_item}
        for word_list_item, target_item in zip(word_list, target)
    ]

    dump_as_jsonl(set, output_dir, filename)
