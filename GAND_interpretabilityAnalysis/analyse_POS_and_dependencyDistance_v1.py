from utils_v1.analyse_POS_and_dependencyDistance_helperFunctions import (
    display_average_pos_chart, find_target_tokens, find_distance, display_dep_freq_chart
)
import argparse
from collections import Counter
import json
import os
import pandas as pd
import spacy
import sys
from typing import Dict


def calculate_average_pos_percentage(
        data: pd.DataFrame, d_saliency_dictionaries: Dict, nlp_spacy: spacy.Language
    ) -> None:
    """Calculate POS tags for each salient token.
    :param data: Complete GAND dataset.
    :param d_saliency_dictionaries: Dictionary containing the saliency dictionaries per target language.
    :param nlp_spacy: spaCy model.
    :return: `None`
    """
    d_all_pos_counts_labeled = {"source": Counter()}
    d_stats = {"source": {"total_found": 0}}

    # Access source sentences to count all POS in EN source
    for _, row in data.iterrows():
        text = row["EN_source_sentence"]

        # Parse the text with spaCy
        doc = nlp_spacy(text)

        # Count all POS tags in the text
        for token in doc:

            if not token.is_space:  # Skip whitespace tokens
                pos_label = token.pos_
                d_all_pos_counts_labeled["source"][pos_label] += 1
                d_stats["source"]["total_found"] += 1

    for target_language in d_saliency_dictionaries:
        salient_dictionary = d_saliency_dictionaries[target_language]
        d_all_pos_counts_labeled[target_language] = Counter()
        d_stats[target_language] = {"total_processed": 0, "total_found": 0, "total_not_found": 0}

        print(f"\n#### Analysis for Salient dictionary for {target_language}: ####")

        for idx, individual_saliency_dict in salient_dictionary.items():

            # Get data from the dictionary
            salient_words = list(source_token for source_token in individual_saliency_dict["salient_data"].keys())
            """salient_scores = list(
                attribution_score[0] for attribution_score in individual_saliency_dict["salient_data"].values()
            )"""
            salient_pos = list(
                attribution_score[1] for attribution_score in individual_saliency_dict["salient_data"].values()
            )
            """attr_difference_words = [
                contr_difference for contr_difference in individual_saliency_dict["prob_data"].keys()
            ]"""
            """attr_difference = [
                contr_difference for contr_difference in individual_saliency_dict["prob_data"].values()
            ]"""

            if not individual_saliency_dict["salient_data"]:
                print(f"No salient data for sentence index {idx} due to neutral gender")
                continue

            d_stats[target_language]["total_processed"] += 1

            # For each salient word, find it in the text and count its POS
            for salient_word, pos in zip(salient_words, salient_pos):

                if salient_word:
                    d_all_pos_counts_labeled[target_language][pos] += 1
                    d_stats[target_language]["total_found"] += 1
                else:
                    d_stats[target_language]["total_not_found"] += 1

    d_pos_percentages = {"source": {}}

    print(f"All POS counts all: {d_all_pos_counts_labeled['source']}")
    print(f"Total overall found: {d_stats['source']}")
    total_overall_found = d_stats["source"]["total_found"]

    for pos in d_all_pos_counts_labeled["source"]:
        labeled_count = d_all_pos_counts_labeled["source"][pos]
        percentage = (labeled_count / total_overall_found) * 100 if total_overall_found > 0 else 0
        d_pos_percentages["source"][pos] = percentage
        print(f"{pos}: {labeled_count}/{total_overall_found} = {percentage:.2f}%\n")

    for target_language in d_saliency_dictionaries:
        print(f"\n=== POS Tags and Frequency for Salient Dictionary for {target_language} ===")
        d_pos_percentages[target_language] = {}

        # Calculate percentages for each POS tag among salient words
        for pos in d_all_pos_counts_labeled[target_language]:
            labeled_count = d_all_pos_counts_labeled[target_language][pos]
            percentage = (labeled_count / d_stats[target_language]["total_found"]) * 100 if d_stats[target_language]["total_found"] > 0 else 0
            d_pos_percentages[target_language][pos] = percentage
            print(f"{pos}: {labeled_count}/{d_stats[target_language]['total_found']} = {percentage:.2f}%")

        print(f"\nSummary:")
        print(f"Total processed for {target_language}: {d_stats[target_language]['total_processed']}")
        print(f"Total salient words found for {target_language}: {d_stats[target_language]['total_found']}")
        print(f"Total POS tags of salient words for {target_language}: {sum(d_all_pos_counts_labeled[target_language].values())}\n")

    # Display results in bar charts
    print("\n")
    target_language_1 = sorted(list(d_saliency_dictionaries.keys()), reverse=True)[0]
    target_language_2 = sorted(list(d_saliency_dictionaries.keys()), reverse=True)[1]
    display_average_pos_chart(
        d_pos_percentages,
        f"POS Tag Distribution of Salient Words (EN-{target_language_1}/{target_language_2})",
        show_percent_symbol=False
    )


def avg_dependencies_distance(data: pd.DataFrame, d_saliency_dictionaries: Dict, nlp_spacy: spacy.Language) -> None:
    """Calculate Dependency distances for all salient words.
    :param data: Complete GAND dataset.
    :param d_saliency_dictionaries: Dictionary containing the saliency dictionaries per target language.
    :param nlp_spacy: spaCy model.
    :return: `None`
    """
    d_all_distances = {"source": []}
    d_freq_neutral_sents = {}

    for idx, row in data.iterrows():
        text = row["EN_source_sentence"]
        referent = row["referent"]
        str_idx = str(idx)

        # Parse the text with spaCy
        doc = nlp_spacy(text)

        target_tokens = find_target_tokens(doc, referent)

        if not target_tokens:
            print(f"Referent word '{referent}' not found in text: {text[:50]} ...")
            continue

        # Collect all non-target tokens (as spaCy Token objects)
        non_target_tokens = []

        for token in doc:  # Check if the current token is one of the target tokens + compare by index for exact token match, and also skip spaces
            is_target_token = False

            for tt in target_tokens:

                if token.i == tt.i:
                    is_target_token = True
                    break

            if not is_target_token and not token.is_space:
                non_target_tokens.append(token)

        # Calculate distances from each non-target token to each target token
        for non_target_token in non_target_tokens:

            for target_tok in target_tokens:
                distance = find_distance(target_tok, non_target_token, doc)

                if distance is not None and distance > 0:  # Only add valid positive distances
                    d_all_distances["source"].append(distance)

        for target_language in d_saliency_dictionaries:
            salient_dictionary = d_saliency_dictionaries[target_language]

            if target_language not in d_all_distances:
                d_all_distances[target_language] = []

            if target_language not in d_freq_neutral_sents:
                d_freq_neutral_sents[target_language] = 0

            # Check if this index exists in salient_dictionary
            if str_idx not in salient_dictionary:
                """print(f"\n=== Skipping Sentence Index {str_idx} (not in salient_dictionary) ===")"""
                continue

            individual_saliency_dict = salient_dictionary[str_idx]

            # Check if there's salient data
            if not individual_saliency_dict.get("salient_data"):
                """print(f"No salient data for sentence index {idx} due to neutral gender")"""
                d_freq_neutral_sents[target_language] += 1
                continue

            # Get data from the dictionary
            salient_words = list(individual_saliency_dict["salient_data"].keys())
            """salient_scores = list(
                attribution_score[0] for attribution_score in individual_saliency_dict["salient_data"].values()
            )"""
            salient_pos = list(
                attribution_score[1] for attribution_score in individual_saliency_dict["salient_data"].values()
            )
            """attr_difference_words = list(individual_saliency_dict["prob_data"].keys())"""
            """attr_difference = list(individual_saliency_dict["prob_data"].values())"""

            # For each salient word, find it in the text and calculate distance
            for salient_word, pos in zip(salient_words, salient_pos):
                """print(f"\tSalient word: '{salient_word}' [{pos}]")"""

                if not salient_word:
                    continue

                # Find the salient word token in the doc
                salient_token = None

                for token in doc:

                    if token.text.lower() == salient_word.lower():
                        salient_token = token
                        break

                # Calculate distance only if both tokens are found
                if salient_token:

                    for target_token in target_tokens:
                        distance = find_distance(target_token, salient_token, doc)

                        if distance is not None and distance > 0:
                            """print(f"\tDistance from '{salient_word}' to '{referent}': {distance}")"""
                            d_all_distances[target_language].append(distance)
                else:
                    """print(f"\t\tSalient word '{salient_word}' not found in text")
                    
                    # Try to find similar tokens
                    similar_tokens = [
                        token.text for token in doc 
                        if salient_word.lower() in token.text.lower() or token.text.lower() in salient_word.lower()
                    ]
                    
                    if similar_tokens:
                        print(f"\t\tSimilar tokens to '{salient_word}': {similar_tokens}")"""

    # Turn the list of all distances into a dictionary with distances and their frequencies
    d_distance_frequency = {lang: Counter(d_all_distances[lang]) for lang in d_all_distances}

    # Calculate average frequency for each distance (dividing by total number of rows)
    total_entries = len(data)
    d_avg_distance_frequency = {
        lang: {distance: count / total_entries for distance, count in d_distance_frequency[lang].items()}
        for lang in d_distance_frequency
    }

    for target_language in d_freq_neutral_sents:
        print(f"Total number of neutral-gender sentences for {target_language}: {d_freq_neutral_sents[target_language]}\n")

    for target_language in d_distance_frequency:
        print(f"\n=== Analysis for {target_language} ===")
        print(f"Total distances calculated for {target_language}: {len(d_all_distances[target_language])}")
        print(f"Distance frequency for {target_language}: {dict(d_distance_frequency[target_language])}")
        print(f"Average distance frequency for {target_language}: {dict(d_avg_distance_frequency[target_language])}\n")

    # Display the chart
    target_language_1 = sorted([k for k in list(d_avg_distance_frequency.keys()) if k != "source"], reverse=True)[0]
    target_language_2 = sorted([k for k in list(d_avg_distance_frequency.keys()) if k != "source"], reverse=True)[1]
    display_dep_freq_chart(
        d_avg_distance_frequency,
        f"Average Dependency Distances & Frequencies (EN-{target_language_1}/{target_language_2})"
    )


def main(languages: str) -> None:
    """Entry point for the script. Analyses POS tags and dependency distance of salient words.
    :param languages: Comma-separated string of the different languages for which the script should be run ('DE' for
        German and 'ES' for Spanish).
    :return: `None`
    """
    # Create output directory if not exists
    if not os.path.isdir(os.path.join("output_v1")):
        os.makedirs(os.path.join("output_v1"))

    # Load spaCy model
    nlp_spacy = spacy.load("en_core_web_lg")

    # Read GAND_contrastive
    gand_contrastive = pd.read_excel(os.path.join("..", "GAND_data", "GAND_CT_sample.xlsx"))

    # Loop over target languages and load previously saved saliency dictionaries
    d_salient_words_per_target_language = {}

    for language in sorted(languages.split(",")):

        # Read the JSONL file back into a dictionary
        jsonl_file_path_1 = os.path.join("output_v1", f"Salient_contrastive_attributions_{language}.jsonl")

        with open(jsonl_file_path_1, "r", encoding="utf-8") as f:
            salient_dictionary = json.load(f)

        d_salient_words_per_target_language[language] = salient_dictionary

    # Calculate POS frequency distribution among salient words
    calculate_average_pos_percentage(gand_contrastive, d_salient_words_per_target_language, nlp_spacy)

    # Calculate dependency distance among salient words
    avg_dependencies_distance(gand_contrastive, d_salient_words_per_target_language, nlp_spacy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "languages", type=str,
        help="Comma-separated string of the different languages for which the script should be run ('DE' for German "
             "and 'ES' for Spanish)."
    )
    main(**vars(parser.parse_args()))
