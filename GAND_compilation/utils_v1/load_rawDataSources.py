import codecs
from datasets import IterableDataset, load_dataset
import os
from pathlib import Path
import random
import sys
from typing import Tuple


def load_c4_data(
        dataset_path: str, subdataset_name: str, dataset_split: str
) -> Tuple[IterableDataset, int]:
    """Load C4 (Common Crawl) data (https://huggingface.co/datasets/allenai/c4) from Hugging Face.
    :param dataset_path: Path to dataset on Hugging Face.
    :param subdataset_name: Name of the subdataset to be loaded.
    :param dataset_split: Split of the subdataset to be loaded.
    :return: The loaded dataset and the number of texts in the dataset.
    """
    dataset = load_dataset(dataset_path, subdataset_name, split=dataset_split, streaming=True)
    len_dataset = dataset.info.splits["train"].num_examples
    print(f"\t- The dataset contains {len_dataset:,d} texts.")

    # Print first item of dataset
    for idx, item in enumerate(dataset):
    
        if idx >= 1:
            break

        example_text_to_be_printed = item['text'][:100].replace("\n", " ")
        print(f"\t- First hundred characters of first text in dataset: '{example_text_to_be_printed} [...]'")

    return dataset, len_dataset


def load_open_subtitles_data(dataset_name: str, n_texts_to_analyse: int) -> Tuple[IterableDataset, int]:
    """Load Open Subtitles data from https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles
    (https://object.pouta.csc.fi/OPUS-OpenSubtitles/v1/mono/en.txt.gz).
    :param dataset_name: Name of the dataset to be filtered.
    :param n_texts_to_analyse: The number of texts from the dataset to analyse.
    :return: The loaded dataset and the number of texts in the dataset.
    """
    if dataset_name == "OpenSubtitles":
        dataset_path = os.path.join("input_v1", "downloadedDataSources", "OpenSubtitles", "en.txt")

    with codecs.open(dataset_path, "r", "utf-8") as f:
        l_lines = [line.strip() for line in f.readlines()]
    f.close()

    len_dataset = len(l_lines)
    print(f"\t- The dataset contains {len_dataset:,d} texts (which usually correspond to single sentences).")
    example_text_to_be_printed = l_lines[0][:100].replace("\n", " ")
    print(f"\t- First hundred characters of first text in dataset: '{example_text_to_be_printed} [...]'")

    def generator(l_items):

        for item in l_items:
            yield {"text": item}

    # Convert list of lines to IterableDataset (cutting off list at `n_texts_to_analyse`)
    random.Random(42).shuffle(l_lines)
    dataset = IterableDataset.from_generator(generator, gen_kwargs={"l_items": l_lines[:n_texts_to_analyse]})

    return dataset, len_dataset
