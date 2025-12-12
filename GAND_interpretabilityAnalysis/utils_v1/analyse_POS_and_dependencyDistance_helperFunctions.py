import matplotlib.pyplot as plt
import numpy as np
import os
from spacy.tokens import Doc, Token
import sys
from typing import Dict, List, Optional


def display_average_pos_chart(d_pos_percentages: Dict, title: str, show_percent_symbol: bool=True) -> None:
    """Display POS tag distribution chart with grouped bars comparing two datasets.
    :param d_pos_percentages: Dictionary containing POS tags and their percentages per target language.
    :param title: Chart title.
    :param show_percent_symbol: Whether to add '%' to bar labels. Defaults to `True`.
    :return: `None`
    """
    d_filtered_data = {}
    l_bools_pos_tags_found = []

    for target_language in d_pos_percentages:
        pos_percentages = d_pos_percentages[target_language]
        d_filtered_data[target_language] = {pos: perc for pos, perc in pos_percentages.items() if perc > 0}

        if not d_filtered_data[target_language]:
            l_bools_pos_tags_found.append(False)
        else:
            l_bools_pos_tags_found.append(True)

    if len(set(l_bools_pos_tags_found)) == 1 and l_bools_pos_tags_found[0] == False:
        print("No POS tags found for salient words")
        return None

    # Get all unique POS tags from both datasets
    all_pos_tags = sorted(set([pos for lang in d_filtered_data for pos in list(d_filtered_data[lang].keys())]))

    # Prepare data for grouped bars
    d_values = {}

    for target_language in d_filtered_data:
        d_values[target_language] = [d_filtered_data[target_language].get(pos, 0) for pos in all_pos_tags]

    # Sort by maximum value (highest value from either dataset)
    target_language_1 = sorted(list(d_values.keys()), reverse=True)[0]
    target_language_2 = sorted(list(d_values.keys()), reverse=True)[1]
    max_values = [max(d_values[target_language_1][i], d_values[target_language_2][i]) for i in range(len(all_pos_tags))]
    sorted_indices = sorted(range(len(max_values)), key=lambda i: max_values[i], reverse=True)

    labels = [all_pos_tags[i] for i in sorted_indices]
    values_1 = [d_values[target_language_1][i] for i in sorted_indices]
    values_2 = [d_values[target_language_2][i] for i in sorted_indices]

    xlabel = "Percentage of Salient Words with this POS Tag (%)"

    # Set up the plot
    plt.figure(figsize=(14, 8))

    # Set bar width and positions
    bar_height = 0.35
    y_pos = np.arange(len(labels))

    # Create grouped horizontal bars
    _ = plt.barh(y_pos - bar_height / 2, values_1, bar_height, label=f"{target_language_1}", color="#1f77b4", alpha=0.8)
    _ = plt.barh(y_pos + bar_height / 2, values_2, bar_height, label=f"{target_language_2}", color="#ff7f0e", alpha=0.8)

    plt.yticks(y_pos, labels)
    plt.title(title, fontsize=32, fontweight="medium")
    plt.xlabel(xlabel, fontsize=32, fontweight="medium")
    plt.ylabel("POS Tags", fontsize=32, fontweight="medium")
    plt.tick_params(axis="both", which="major", labelsize=25)
    plt.legend(fontsize=32, loc="upper right")

    plt.tight_layout()
    plt.grid(axis="x", alpha=0.3)

    safe_filename = os.path.join("output_v1", title.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", ""))
    plt.savefig(f"{safe_filename}.png", dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {safe_filename}.png")
    """plt.show()"""

    return None


def get_path_to_root(token_idx: int, token_to_head: Dict) -> List:
    """Find the path from a word to the root.
    :param token_idx: Index of the token.
    :param token_to_head: Dictionary mapping tokens to their heads.
    :return: A list containing the path to the root.
    """
    path = []
    visited = set()  # Prevent infinite loops

    while token_idx != token_to_head[token_idx] and token_idx not in visited:
        visited.add(token_idx)
        path.append(token_idx)
        token_idx = token_to_head[token_idx]

    path.append(token_idx)

    return path


def find_distance(word1: Token, word2: Token, doc: Doc) -> Optional[int]:
    """Calculate distance in the dependency tree.
    :param word1: Referent token.
    :param word2: Salient token.
    :param doc: spaCy doc.
    :return: The dependency distance between the referent token and salient token.
    """
    # Create a mapping of token index to its head (parent) index
    token_to_head = {token.i: token.head.i for token in doc}

    # Get the paths from both words to the root
    path1 = get_path_to_root(word1.i, token_to_head)
    path2 = get_path_to_root(word2.i, token_to_head)

    # Find the common ancestor
    common_ancestor = None

    for token in path1:

        if token in path2:
            common_ancestor = token
            break

    # Calculate distance only if common ancestor is found in both paths
    if common_ancestor is not None and common_ancestor in path1 and common_ancestor in path2:
        distance = path1.index(common_ancestor) + path2.index(common_ancestor)
        return distance
    else:
        # Handle case where common ancestor is not found
        return None


def display_dep_freq_chart(d_avg_distance_frequency: Dict, title: str) -> None:
    """Display dependency distance distribution chart with grouped bars comparing two datasets.
    :param d_avg_distance_frequency: Dictionary containing average distance frequencies.
    :param title: Chart title.
    :return: `None`
    """
    # Filter out 0 and `None` from both datasets
    d_filtered_data = {}
    l_bools_filtered_data = []

    for target_language in d_avg_distance_frequency:
        distance_frequency = d_avg_distance_frequency[target_language]
        d_filtered_data[target_language] = {
            label: distance_frequency[label] for label in distance_frequency if label and label != 0
        }

        if not d_filtered_data[target_language]:
            l_bools_filtered_data.append(False)
        else:
            l_bools_filtered_data.append(True)

    if len(set(l_bools_filtered_data)) == 1 and l_bools_filtered_data[0] == False:
        print("No valid distances to plot")
        return None

    # Get all unique distances from both datasets
    all_distances = sorted(
        set([distance for lang in d_filtered_data for distance in list(d_filtered_data[lang].keys())])
    )

    # Prepare data for grouped bars
    target_language_1 = sorted(list(d_filtered_data.keys()), reverse=True)[0]
    target_language_2 = sorted(list(d_filtered_data.keys()), reverse=True)[1]
    values_1 = [d_filtered_data[target_language_1].get(dist, 0) for dist in all_distances]
    values_2 = [d_filtered_data[target_language_2].get(dist, 0) for dist in all_distances]

    # Set up the plot
    plt.figure(figsize=(14, 7))

    # Set bar width and positions
    bar_width = 0.35
    x_pos = np.arange(len(all_distances))

    # Create grouped bars
    _ = plt.bar(x_pos - bar_width / 2, values_1, bar_width, label=f"{target_language_1}", color="#1f77b4", alpha=0.8)
    _ = plt.bar(x_pos + bar_width / 2, values_2, bar_width, label=f"{target_language_2}", color="#ff7f0e", alpha=0.8)

    plt.xticks(x_pos, all_distances)
    plt.title(title, fontsize=32, fontweight="medium")
    plt.xlabel("Distances in the Dependency Tree", fontsize=32, fontweight="medium")
    plt.ylabel("Average Frequency", fontsize=32, fontweight="medium")
    plt.legend(fontsize=28, loc="upper right")

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    safe_filename = os.path.join("output_v1", title.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", ""))
    plt.savefig(f"{safe_filename}.png", dpi=300, bbox_inches="tight")
    print(f"Plot saved as: {safe_filename}.png")
    """plt.show()"""

    return None
