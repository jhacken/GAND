import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
from typing import Dict


def plot_visualisations(d_prob_difference_dfs: Dict) -> None:
    """Create visualisations.
    :param d_prob_difference_dfs: Dictionary containing dataframes with probability differences per target language.
    :return: `None`
    """
    # Create combined dataset for comparison
    l_dfs_copy = []

    for target_language in d_prob_difference_dfs:
        df = d_prob_difference_dfs[target_language]
        df_copy = df.copy()
        df_copy["language"] = target_language
        df_copy["gender"] = df[f"{target_language}_gender"]
        l_dfs_copy.append(df_copy)

    combined_df = pd.concat([df[["gender", "probability", "language"]] for df in l_dfs_copy])

    # Combined boxplot
    lang_1 = sorted(list(d_prob_difference_dfs.keys()), reverse=True)[0]
    lang_2 = sorted(list(d_prob_difference_dfs.keys()), reverse=True)[1]
    fig, ax = plt.subplots(figsize=(12, 6))
    combined_df.boxplot(column="probability", by=["gender", "language"], ax=ax)
    ax.set_title(f"Probability Distribution: {lang_1} vs {lang_2}")
    ax.set_xlabel("Gender and Language")
    ax.set_ylabel("Probability")
    plt.suptitle("")  # Remove default title

    plt.tight_layout()
    save_filename = os.path.join("output_v1", f"Probability_Distribution_EN-{lang_1}-{lang_2}_boxplot.png")
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    print(f"Boxplot saved as: {save_filename}")
    """plt.show()"""
    print("\n")

    # Side-by-side violinplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.violinplot(data=d_prob_difference_dfs[lang_1], x=f"{lang_1}_gender", y="probability", ax=ax1)
    ax1.set_title(f"{lang_1} - Probability by Gender")
    ax1.set_xlabel("Gender")
    ax1.set_ylabel("Probability")

    sns.violinplot(data=d_prob_difference_dfs[lang_2], x=f"{lang_2}_gender", y="probability", ax=ax2)
    ax2.set_title(f"{lang_2} - Probability by Gender")
    ax2.set_xlabel("Gender")
    ax2.set_ylabel("Probability")

    plt.tight_layout()
    save_filename = os.path.join("output_v1", f"Probability_Distribution_EN-{lang_1}-{lang_2}_violinplot.png")
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    print(f"Violinplot saved as: {save_filename}")
    """plt.show()"""
    print("\n")

    # Combined comparison plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=combined_df, x="gender", y="probability", hue="language", split=False, ax=ax)
    ax.set_title(f"Probability Distribution Comparison: {lang_1} vs {lang_2}", fontsize=28, fontweight="medium")
    ax.set_xlabel("Gender", fontsize=28, fontweight="medium")
    ax.set_ylabel("Probability", fontsize=28, fontweight="medium")
    ax.tick_params(axis="both", which="major", labelsize=22)  # Increase tick label size
    ax.legend(fontsize=20, loc="best")  # Increase legend font size

    save_filename = os.path.join("output_v1", f"Probability_Comparison_{lang_1}_vs_{lang_2}.png")
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")
    print(f"Comparison plot saved as: {save_filename}")
    """plt.show()"""
