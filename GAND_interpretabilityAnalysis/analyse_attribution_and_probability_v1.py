from utils_v1.analyse_attribution_and_probability_helperFunctions import (
    plot_referent_dotplot, plot_visualisations_prob_difference, plot_visualisations_salient_words_combined
)
import argparse
from collections import Counter
import copy
import json
import os
import pandas as pd
import sys
from typing import Dict, List, Tuple


def analyse_salient_words(
        salient_dictionary: Dict, target_language: str
) -> Tuple[List, List, List, List, List, pd.DataFrame]:
    """Analyse salient words.
    :param salient_dictionary: Dictionary containing salient words.
    :param target_language: The target language.
    :return: A tuple containing a list of all salient words, a list of salient words found two, three, four, and 5+
        times, respectively, and a dataframe containing all saliency scores.
    """
    all_salient_words = []
    all_saliency_scores = []

    for index, individual_saliency_dict in salient_dictionary.items():
        salient_words = list(source_token for source_token in individual_saliency_dict["salient_data"].keys())
        all_salient_words.extend(salient_words)
        salient_scores = list(score for score, pos in individual_saliency_dict["salient_data"].values())
        all_saliency_scores.extend(salient_scores)

    print(f"###### Salient duplicates for EN-{target_language} ######")
    """print(f"All salient words: {all_salient_words}")"""
    print(f"Number of salient words: {len(all_salient_words)}")

    # Use Counter to count the occurrences of each element in the list
    counts = Counter(all_salient_words)
    duplicates_2 = [item for item, count in counts.items() if count == 2]
    duplicates_3 = [item for item, count in counts.items() if count == 3]
    duplicates_4 = [item for item, count in counts.items() if count == 4]
    duplicates_5 = [item for item, count in counts.items() if count > 4]
    print(f"Duplicates found 2x among all salient words: {duplicates_2}")
    print(f"Duplicates found 3x among all salient words: {duplicates_3}")
    print(f"Duplicates found 4x among all salient words: {duplicates_4}")
    print(f"Duplicates found 5+ times among all salient words: {duplicates_5}")
    print(f"Number of salient words appearing twice: {len(duplicates_2)}")
    print(f"Number of salient words appearing 3x: {len(duplicates_3)}")
    print(f"Number of salient words appearing 4x: {len(duplicates_4)}")
    print(f"Number of salient words appearing 5+ times: {len(duplicates_5)}")
    print(f"Percentage of salient words appearing twice (based on all salient words): "
          f"{((len(duplicates_2) / len(all_salient_words))*100):.2f}%")
    print(f"Percentage of salient words appearing 3x (based on all salient words): "
          f"{((len(duplicates_3) / len(all_salient_words))*100):.2f}%")
    print(f"Percentage of salient words appearing 4x (based on all salient words): "
          f"{((len(duplicates_4) / len(all_salient_words))*100):.2f}%")
    print(f"Percentage of salient words appearing 5+ times (based on all salient words): "
          f"{((len(duplicates_5) / len(all_salient_words))*100):.2f}%")

    # Get statistics of salient scores
    print(f"\n##### Statistics of salient scores (EN-{target_language}) #####")
    all_saliency_scores_df = pd.DataFrame(all_saliency_scores)
    print(all_saliency_scores_df.describe())

    return all_salient_words, duplicates_2, duplicates_3, duplicates_4, duplicates_5, all_saliency_scores_df


def identify_overlapping_salient_words(d_saliency_dictionaries: Dict) -> Dict:
    """Identify overlapping salient words in the different target languages.
    :param d_saliency_dictionaries: Dictionary containing the saliency dictionaries per target language.
    :return: A dictionary containing the overlapping salient words.
    """
    d_all_saliency_words = {}
    d_all_saliency_pos = {}

    for target_language in d_saliency_dictionaries:
        saliency_dictionary = d_saliency_dictionaries[target_language]
        d_all_saliency_words[target_language] = []
        d_all_saliency_pos[target_language] = []

        for index, individual_saliency_dict in saliency_dictionary.items():
            salient_words = list(source_token for source_token in individual_saliency_dict["salient_data"].keys())
            d_all_saliency_words[target_language].extend(salient_words)
            salient_pos = list(pos for score, pos in individual_saliency_dict["salient_data"].values())
            d_all_saliency_pos[target_language].extend(salient_pos)

        print(f"Salient words for EN-{target_language}:\n{d_all_saliency_words[target_language]}")
        print(f"Salient words for EN-{target_language}: {len(d_all_saliency_words[target_language])}\n")

    overlapping_salient_words = {}
    target_language_1 = sorted(list(d_all_saliency_words.keys()))[0]
    target_language_2 = sorted(list(d_all_saliency_words.keys()))[1]

    for salient_word, salient_pos in zip(d_all_saliency_words[target_language_1], d_all_saliency_pos[target_language_1]):

        if salient_word in d_all_saliency_words[target_language_2]:
            overlapping_salient_words[salient_word] = salient_pos

    print("##### Salient words both in EN-DE and in EN-ES #####")
    print(overlapping_salient_words)
    print(f"Number of overlapping salient words EN-DE/ES: {len(overlapping_salient_words)}")

    for target_language in d_all_saliency_words:
        print(f"Percentage of overlapping salient words EN-DE/ES w.r.t. {target_language}: "
              f"{(len(overlapping_salient_words.keys()) / len(d_all_saliency_words[target_language]) * 100):.2f}")

    return overlapping_salient_words


def get_prob_difference_and_salient_words(data: pd.DataFrame, d_saliency_dictionaries: Dict) -> Dict:
    """Get probability differences.
    :param data: Complete GAND dataset.
    :param d_saliency_dictionaries: Dictionary containing the saliency dictionaries per target language.
    :return: A dictionary containing the probability differences in a pandas DataFrame.
    """
    d_prob_difference_and_salient_word_lists = {}

    for target_language in d_saliency_dictionaries:
        d_prob_difference_and_salient_word_lists[target_language] = []
        saliency_dict = d_saliency_dictionaries[target_language]

        for idx, row in data.iterrows():
            """text = row["EN_source_sentence"]"""
            referent = row["referent"]
            referent_embedding = row["referent_embedding"]
            """contrastive_gender = row[f"{target_language}_contrastive_gender"]"""
            gender = row[f"{target_language}_gender"]

            # Convert idx to string to match saliency_dict keys
            str_idx = str(idx)

            if str_idx in saliency_dict:
                individual_saliency_dict = saliency_dict[str_idx]

                if individual_saliency_dict["salient_data"]:  # Only skip if no salient data
                    prob_value = list(individual_saliency_dict["prob_data"].values())[0]
                    token_diff = list(individual_saliency_dict["prob_data"].keys())[0]
                    salient_word = list(individual_saliency_dict["salient_data"].keys())[0]
                    salient_pos = list(individual_saliency_dict["salient_data"].values())[0][1]

                    # Append row data as a dictionary
                    d_prob_difference_and_salient_word_lists[target_language].append({
                        "index": idx,
                        "referent": referent,
                        "referent_embedding": referent_embedding,
                        f"{target_language}_gender": gender,
                        "probability": prob_value,
                        "token_diff": token_diff,
                        "salient_words": salient_word,
                        "salient_pos": salient_pos
                    })

    # Create DataFrames from list of dictionaries
    d_prob_difference_and_salient_words_dfs = {
        lang: pd.DataFrame(d_prob_difference_and_salient_word_lists[lang]) 
        for lang in d_prob_difference_and_salient_word_lists
    }
    d_prob_difference_dfs_overall = copy.deepcopy(d_prob_difference_and_salient_words_dfs)
    d_prob_difference_dfs_per_referent = copy.deepcopy(d_prob_difference_and_salient_words_dfs)
    d_salient_words_dfs = copy.deepcopy(d_prob_difference_and_salient_words_dfs)

    # We want to work with only "masculine" and "feminine", as for neutral there should be no contrastive difference
    for target_language in d_prob_difference_dfs_overall:
        df_upd = copy.deepcopy(d_prob_difference_dfs_overall[target_language])
        df_upd = df_upd[df_upd[f"{target_language}_gender"].isin(["masculine", "feminine"])]
        d_prob_difference_dfs_overall[target_language] = df_upd

    # Print statistics and create visualisations (overall, for probability differences)
    proceed_with_visualisation = True

    for target_language in d_prob_difference_dfs_overall:
        prob_difference_df = d_prob_difference_dfs_overall[target_language]
        print(f"\n\n=== {target_language} Statistics ===\n")

        if not prob_difference_df.empty:
            gender_stats_1 = prob_difference_df.groupby(f"{target_language}_gender")["probability"].describe()
            print(gender_stats_1)
            mean_by_gender_1 = prob_difference_df.groupby(f"{target_language}_gender")["probability"].mean()
            print("\nMean probability by gender:")
            print(mean_by_gender_1, "\n")
        else:
            print("No data available\n")
            proceed_with_visualisation = False

    if proceed_with_visualisation:
        plot_visualisations_prob_difference(d_prob_difference_dfs_overall)
    else:
        print("Cannot create plots - one or both dataframes are empty")

    # Print statistics and create visualisations (per referent, for probability differences)
    embedding_mapping = {
        "LLM_female_list": "fem_embedding", "female_embedding_list": "fem_embedding",
        "LLM_male_list": "masc_embedding", "male_embedding_list": "masc_embedding",
        "LLM_neutral_list": "neut_embedding"
    }

    for target_language in d_prob_difference_dfs_per_referent:
        df = d_prob_difference_dfs_per_referent[target_language]
        df_summary_per_referent = df.groupby(["referent", "referent_embedding"]).agg(
            gender=(f"{target_language}_gender", list),
            probability=("probability", list),
            mean_probability=("probability", "mean"),
            std_probability=("probability", "std")
        ).reset_index()
        df_summary_per_referent["majority_gender"] = df_summary_per_referent["gender"].apply(
            lambda x: max(set(x), key=x.count)
        )
        df_summary_per_referent = df_summary_per_referent.rename(columns={"gender": f"{target_language}_gender"})
        df_summary_per_referent["referent_embedding"] = df_summary_per_referent["referent_embedding"].map(embedding_mapping)

        neutral_referents = df_summary_per_referent[df_summary_per_referent[f"{target_language}_gender"].apply(
            lambda x: all(g == "neutral" for g in x)
        )]

        if not neutral_referents.empty:
            neutral_referents_upd = neutral_referents.copy()
            neutral_referents_upd["majority_gender"] = "neutral"
            neutral_referents_upd["mean_probability"] = 0.0
            neutral_referents_upd["std_probability"] = 0.0

            # Keep only non-neutral rows in the original summary, then append neutral ones
            non_neutral = df_summary_per_referent[~df_summary_per_referent.index.isin(neutral_referents_upd.index)]
            df_summary_per_referent = pd.concat([non_neutral, neutral_referents_upd]).reset_index(drop=True)

        print(f"=== {target_language} Referent Summary ===")
        print(df_summary_per_referent)
        
        """df_summary_per_referent.to_excel(os.path.join("output_v1", f"referent_summary_{target_language}.xlsx"))"""
        plot_referent_dotplot(df_summary_per_referent, target_language)

    # Print statistics and create visualisations (per referent, for salient words)
    d_summary_salient_words_dfs = {}

    for target_language in d_salient_words_dfs:
        df = d_salient_words_dfs[target_language]
        df_summary_per_referent = df.groupby(["referent", "referent_embedding"]).agg(
            gender=(f"{target_language}_gender", list),
            salient_words=("salient_words", list),
            salient_pos=("salient_pos", list),
        ).reset_index()
        df_summary_per_referent["majority_gender"] = df_summary_per_referent["gender"].apply(
            lambda x: max(set(x), key=x.count)
        )
        df_summary_per_referent = df_summary_per_referent.rename(columns={"gender": f"{target_language}_gender"})
        d_summary_salient_words_dfs[target_language] = df_summary_per_referent

        print(f"=== {target_language} Referent Summary ===")
        print(df_summary_per_referent)

    plot_visualisations_salient_words_combined(d_summary_salient_words_dfs)

    return d_prob_difference_and_salient_words_dfs


def main(languages: str) -> None:
    """Entry point for the script. Analyses attribution and probability scores.
    :param languages: Comma-separated string of the different languages for which the script should be run ('DE' for
        German and 'ES' for Spanish).
    :return: `None`
    """
    # Create output directory if not exists
    if not os.path.isdir(os.path.join("output_v1")):
        os.makedirs(os.path.join("output_v1"))

    # Read GAND_contrastive
    gand_contrastive = pd.read_excel(os.path.join("..", "GAND_data" ,"GAND_CT_sample.xlsx"))

    # Loop over target languages, load previously saved saliency dictionaries, and perform analysis
    d_salient_words_per_target_language = {}

    for language in sorted(languages.split(",")):
        print(f"\n========== Running script for {language} ==========\n\n")

        # Read the JSONL file back into a dictionary
        jsonl_file_path_1 = os.path.join("output_v1", f"Salient_contrastive_attributions_{language}.jsonl")

        with open(jsonl_file_path_1, "r", encoding="utf-8") as f:
            salient_dictionary = json.load(f)

        d_salient_words_per_target_language[language] = salient_dictionary

        _, _, _, _, _, _ = analyse_salient_words(salient_dictionary, language)

    # Identify overlapping salient words
    _ = identify_overlapping_salient_words(d_salient_words_per_target_language)

    # Get probability differences and salient words + create visualisations
    _ = get_prob_difference_and_salient_words(gand_contrastive, d_salient_words_per_target_language)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "languages", type=str,
        help="Comma-separated string of the different languages for which the script should be run ('DE' for German "
             "and 'ES' for Spanish)."
    )
    main(**vars(parser.parse_args()))
