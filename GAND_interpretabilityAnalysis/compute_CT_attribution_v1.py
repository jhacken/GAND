from utils_v1.compute_CT_attribution_helperFunctions import (
    save_all_normalized_merged_source_attributions_for_first_arrow_word, get_sums_scores_per_sentence
)
import argparse
import inseq
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import spacy
import sys
from tqdm import tqdm
from typing import Dict, List, Tuple, Union


def open_gand_xlsx(path_to_xlsx: Union[str, Path], target_language: str) -> Tuple[List, List, List, pd.DataFrame, int]:
    """Open the file containing the GAND dataset and run some basic statistics.
    :param path_to_xlsx: Path to GAND dataset.
    :param target_language: The target language.
    :return: A tuple containing the list of source sentences, the list of translations, the list of contrastive
        translations, the complete GAND dataset, and the length of the dataset.
    """
    # Read the tab-separated text into a pandas DataFrame
    gand_contrastive = pd.read_excel(path_to_xlsx)

    # Access columns and save contents to a list
    en_source = gand_contrastive["EN_source_sentence"].tolist()
    dataset_length = len(en_source)
    opus_mt = gand_contrastive[f"OPUS_{target_language}_translation"].tolist()
    opus_mt_contrastive = gand_contrastive[f"OPUS_{target_language}_Contrastive"].tolist()

    print("\n=== GAND EXAMPLE ===")
    print(f"EN Source: {en_source[0]}")
    print(f"OPUS-MT {target_language} Translation: {opus_mt[0]}")
    print(f"OPUS-MT {target_language} Contrastive Translation: {opus_mt_contrastive[0]}")

    #  Calculate lengths, filter out NaN and keep only valid strings
    sentence_lengths = [len(sentence.split()) for sentence in en_source if isinstance(sentence, str)]

    # Calculate statistics
    average_length = np.mean(sentence_lengths)
    std_length = np.std(sentence_lengths, ddof=1)  # ddof=1 for sample std deviation

    # Print results
    print("\n=== EN SENTENCE LENGTH STATISTICS ===")
    print(f"Number of source sentences: {len(en_source)}")
    print(f"Average length: {average_length:.2f} words")
    print(f"Standard deviation: {std_length:.2f} words")
    print(f"Min length: {min(sentence_lengths)} words")
    print(f"Max length: {max(sentence_lengths)} words")
    print(f"Total sentences: {len(sentence_lengths)}")
    print(f"Sentences with NaN values skipped: {len(en_source) - len(sentence_lengths)}\n")

    return en_source, opus_mt, opus_mt_contrastive, gand_contrastive, dataset_length


def compute_all_attribution_saliency_scores(
        target_language: str, en_source: List, opus_mt: List, opus_mt_contrastive: List, gand_contrastive: pd.DataFrame,
        dataset_length: int, attribution_model: inseq.AttributionModel, nlp_spacy: spacy.Language
) -> Dict:
    """Compute all attribution saliency scores and save them to a jsonl file.
    :param target_language: The target language.
    :param en_source: List of source sentences.
    :param opus_mt: List of original Opus MT translations.
    :param opus_mt_contrastive: List of contrastive Open MT translations.
    :param gand_contrastive: Complete GAND dataset.
    :param dataset_length: Length of the dataset.
    :param attribution_model: inseq attribution model.
    :param nlp_spacy: spaCy model.
    :return: A dictionary containing the attribution saliency scores.
    """
    # Create an empty dictionary to collect previously combined dictionaries
    all_contrastive_attributions = {}

    for i in tqdm(range(dataset_length), desc="Computing  all attribution saliency scores ..."):
        en = en_source[i]
        tgt_orig = opus_mt[i]
        tgt_cont = opus_mt_contrastive[i]

        # Gets original tokens in format [(0, '</s>'), (1, 'deu_Latn'), (2, '▁Beispi'), (3, 'els'), (4, 'atz'), (5, '▁auf'), (6, '▁Eng'), (7, 'lis'), (8, 'ch'), (9, ','), (10, '▁um'), (11, '▁zu'), (12, '▁zeigen'), (13, ','), (14, '▁wes'), (15, 'halb'), (16, '▁dies'), (17, '▁ein'), (18, '▁Problem'), (19, '▁für'), (20, '▁den'), (21, '▁Program'), (22, 'mi'), (23, 'erer'), (24, '▁ist'), (25, '.'), (26, '</s>')]
        orig_tokens = list(enumerate(attribution_model.encode(tgt_orig, as_targets=True).input_tokens[0]))
        out = attribution_model.attribute(
            en,
            tgt_orig,
            contrast_targets=tgt_cont,
            attribute_target=True,
            attributed_fn="contrast_prob_diff",
            step_scores=["contrast_prob_diff"],
            contrast_targets_alignments=[(i,i) for i in range(len(orig_tokens))],
        )

        """print(f"\n=== Processing Sentence {i+1} ===")"""
        token_attribution_and_probs = save_all_normalized_merged_source_attributions_for_first_arrow_word(
            out, gand_contrastive, i, nlp_spacy
        )

        # To final dictionary, add index as key and contrastive prob difference + tokens attributions (and POS) as
        # values
        all_contrastive_attributions[i] = token_attribution_and_probs

    """print(f"All contrastive attributions dict: {all_contrastive_attributions}")"""

    # Save dict as JSONL file
    jsonl_file_path = os.path.join("output_v1", f"All_contrastive_attributions_{target_language}.jsonl")

    with open(jsonl_file_path, "w", encoding="utf-8") as f:
        json_str = json.dumps(all_contrastive_attributions, ensure_ascii=False, default=str)
        f.write(json_str + '\n')

    print(f"\nSaved to {jsonl_file_path}")

    return all_contrastive_attributions


def get_top_salient_words(
        all_contrastive_attributions: Dict, target_language: str,
        *,
        percent: float = 0.2
) -> Dict:
    """Threshold function to select top salient words.
    :param all_contrastive_attributions: Attribution dictionary.
    :param target_language: The target language.
    :param percent: Top x% of salient words chosen based on total attribution value per length. Defaults to 20%.
    :return: A dictionary containing the top salient words and their corresponding values.
    """
    # Stopwords
    stopwords = ["", "a", "an", "the", "this", "that", "these", "those"]

    # Run processing code
    salient_dictionary = get_sums_scores_per_sentence(all_contrastive_attributions, stopwords, percent)

    # Save dict as JSONL file
    jsonl_file_path = os.path.join("output_v1", f"Salient_contrastive_attributions_{target_language}.jsonl")

    with open(jsonl_file_path, "w", encoding="utf-8") as f:
        json_str = json.dumps(salient_dictionary, ensure_ascii=False, default=str)
        f.write(json_str + '\n')

    print(f"\nSaved to {jsonl_file_path}")

    return salient_dictionary


def main(languages: str) -> None:
    """Entry point for the script. Computes the contrastive translation attribution scores using inseq.
    :param languages: Comma-separated string of the different languages for which the script should be run ('DE' for
        German and 'ES' for Spanish).
    :return: `None`
    """
    # Create output directory if not exists
    if not os.path.isdir(os.path.join("output_v1")):
        os.makedirs(os.path.join("output_v1"))

    # Load spaCy model
    nlp_spacy = spacy.load("en_core_web_lg")

    # Loop over different languages to compute the contrastive translation attribution scores
    for target_language in sorted(languages.split(",")):
        print(f"\n========== Running script for {target_language} ==========\n\n")

        if target_language == "DE":
            opus_mt_model = "Helsinki-NLP/opus-mt-en-de"

        if target_language == "ES":
            opus_mt_model = "Helsinki-NLP/opus-mt-en-es"

        # Open the GAND dataset
        en_source, opus_mt, opus_mt_contrastive, gand_contrastive, dataset_length = open_gand_xlsx(
            os.path.join("..", "GAND_data" ,"GAND_CT_sample.xlsx"), target_language
        )

        # Load the Attribution Model: we chose Helsinki's OPUS-MT en-de / en-es (same as used for the translations), and we focus on 'saliency'
        attribution_model = inseq.load_model(opus_mt_model, "saliency")

        # Compute all attribution saliency scores and save them to a jsonl file
        all_contrastive_attributions = compute_all_attribution_saliency_scores(
            target_language, en_source, opus_mt, opus_mt_contrastive, gand_contrastive, dataset_length,
            attribution_model, nlp_spacy
        )

        # Call threshold function to select top salient words
        _ = get_top_salient_words(all_contrastive_attributions, target_language)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "languages", type=str,
        help="Comma-separated string of the different languages for which the script should be run ('DE' for German "
             "and 'ES' for Spanish)."
    )
    main(**vars(parser.parse_args()))
