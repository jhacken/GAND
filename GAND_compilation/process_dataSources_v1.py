from utils_v1.filter_data_coreference import filter_coreference
from utils_v1.filter_data_referentEntities import filter_referent_entities
from utils_v1.load_rawDataSources import load_c4_data, load_open_subtitles_data
from utils_v1.process_resources import load_list_of_referents
from utils_v1.write_output import merge_data_per_item_into_single_xlsx, sentences_filtered_out_to_txt
import argparse
from datasets import IterableDataset
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple


def perform_step_1() -> Tuple[List, Dict]:
    print("STEP_1: Define list of referent entities")
    list_of_referents, d_referents_to_source = load_list_of_referents(
        os.path.join("input_v1", "predefinedWordLists")
    )
    
    return list_of_referents, d_referents_to_source


def perform_step_2(dataset_name: str, n_texts_to_analyse: int) -> Tuple[IterableDataset, int]:
    print(f"\n-----------\n\nSTEP_2: Load raw data of {dataset_name} dataset")

    if dataset_name == "C4":
        dataset, len_dataset = load_c4_data("allenai/c4", "en", "train")

    if dataset_name == "OpenSubtitles":
        dataset, len_dataset = load_open_subtitles_data(n_texts_to_analyse)
    
    return dataset, len_dataset


def perform_step_3(
        timestr: str, dataset_name: str, list_of_referents: List, dataset: IterableDataset, len_dataset: int,
        n_texts_to_analyse: int, min_sentence_length: int, max_sentence_length: int
) -> Dict:
    print("\n-----------\n\nSTEP_3: Filter dataset (extract sentences containing the referent entities)")
    dict_sentences_per_referent = filter_referent_entities(
        timestr, dataset_name, list_of_referents, dataset, len_dataset,
        n_texts_to_analyse, min_sentence_length, max_sentence_length
    )

    return dict_sentences_per_referent


def perform_step_4(
        timestr: str, dataset_name: str, dict_sentences_per_referent: Dict, device: str,
        max_n_sentences_per_referent: Optional[int]
) -> Dict:
    print("\n-----------\n\nSTEP_4: Filter dataset (exclude unsuitable sentences and sentences including "
          "co-reference) and, if indicated, write filtered dataset to XLSX files per referent entity")
    dict_sentences_filtered_out = filter_coreference(
        timestr, dataset_name, dict_sentences_per_referent, device, max_n_sentences_per_referent
    )

    return dict_sentences_filtered_out


def perform_step_5(
        timestr: str, dataset_name: str, list_of_referents: List, dict_sentences_filtered_out: Dict,
        d_referents_to_source: Dict, d_meta: Dict
) -> None:
    print("\n-----------\n\nSTEP_5: Write sentences that were filtered out to TXT files, join data per referent "
          "entity into one single overarching file, and delete temporary files")
    sentences_filtered_out_to_txt(
        os.path.join("output_v1", "sentences_filtered_out", f"{dataset_name}_{timestr}"), dict_sentences_filtered_out
    )
    merge_data_per_item_into_single_xlsx(
        timestr, dataset_name, list_of_referents, d_referents_to_source, d_meta,
        os.path.join("temp_v1", "sentences_selected_perItem", f"{dataset_name}_{timestr}"),
        os.path.join("output_v1", "sentences_selected_joined")
    )
    shutil.rmtree(os.path.join("temp_v1"))


def main(
        dataset_name: str, device: str, n_texts_to_analyse: int, min_sentence_length: int, max_sentence_length: int,
        max_n_sentences_per_referent: str
) -> None:
    """Entry point of the data filtering script.
    :param dataset_name: Name of the data source to be filtered. Choose between: 'C4' and 'OpenSubtitles'.
    :param device: Device on which the script should be run. Defaults to 'cpu'.
    :param n_texts_to_analyse: Number of texts from the dataset to be analysed. Defaults to 10,000.
    :param min_sentence_length: Minimum number of tokens a sentence should contain. All sentences that do not meet this
        threshold are excluded. Defaults to 10.
    :param max_sentence_length: Maximum number of tokens a sentence can contain. All sentences that exceed this
        threshold are excluded. Defaults to 50.
    :param max_n_sentences_per_referent: Maximum number of sentences per referent entity (i.e. person) to be selected.
        'None' if all sentences should be selected. Defaults to 500.
    :return: `None`
    """
    if max_n_sentences_per_referent == "None":
        max_n_sentences_per_referent = None
    else:
        max_n_sentences_per_referent = int(max_n_sentences_per_referent)

    # DEFINE TIME STRING OF CURRENT JOB
    timestr = time.strftime("%Y%m%d-%H%M%S")
    start_time = time.time()

    # STEP_1: Define list of referent entities
    list_of_referents, d_referents_to_source = perform_step_1()

    # STEP_2: Load dataset
    dataset, len_dataset = perform_step_2(dataset_name, n_texts_to_analyse)

    # STEP_3: Filter dataset (extract sentences containing the referent entities)
    dict_sentences_per_referent = perform_step_3(
        timestr, dataset_name, list_of_referents, dataset, len_dataset, n_texts_to_analyse, min_sentence_length, max_sentence_length
    )
    
    # STEP_4: Filter dataset (exclude sentences including co-reference) and, if indicated, write filtered dataset to XLSX files 
    # per referent entity
    dict_sentences_filtered_out = perform_step_4(
        timestr, dataset_name, dict_sentences_per_referent, device, max_n_sentences_per_referent
    )

    # STEP_5: Write sentences that were filtered out to TXT files, join data per referent entity into one single overarching file, 
    # and delete temporary files
    d_meta = {
        "n_texts_to_analyse": n_texts_to_analyse,
        "min_sentence_length": min_sentence_length,
        "max_sentence_length": max_sentence_length,
        "max_n_sentences_per_referent": max_n_sentences_per_referent
    }
    perform_step_5(timestr, dataset_name, list_of_referents, dict_sentences_filtered_out, d_referents_to_source, d_meta)

    # Print runtime of the script
    end_time = time.time()
    print(f"\n-----------\n\nIt took the script {round(((end_time - start_time) / 60), 2)} minutes to run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "dataset_name",
        help="Name of the data source to be processed.",
        choices=["C4", "OpenSubtitles"]
    )
    parser.add_argument(
        "-d", "--device",
        default="cpu", type=str,
        help="Device on which the script should be run. Defaults to 'cpu'."
    )
    parser.add_argument(
        "-t", "--n_texts_to_analyse",
        default=10000, type=int,
        help="Number of texts from the dataset to be analysed. Defaults to 10,000."
    )
    parser.add_argument(
        "-minsl", "--min_sentence_length",
        default=10, type=int,
        help="Minimum number of tokens a sentence should contain. All sentences that do not meet this threshold are "
             "excluded. Defaults to 10."
    )
    parser.add_argument(
        "-maxsl", "--max_sentence_length",
        default=50, type=int,
        help="Maximum number of tokens a sentence can contain. All sentences that exceed this threshold are excluded. "
             "Defaults to 50."
    )
    parser.add_argument(
        "-sr", "--max_n_sentences_per_referent",
        default="500", type=str,
        help="Maximum number of sentences per referent entity (i.e. person) to be selected. 'None' if all sentences "
             "should be selected. Defaults to 500."
    )
    main(**vars(parser.parse_args()))
