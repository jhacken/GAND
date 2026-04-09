from collections import Counter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import sys
from typing import Dict, List, Tuple
from wordcloud import WordCloud


def plot_referent_dotplot(df_summary: pd.DataFrame, target_language: str) -> None:
    """Plot dotplot for probability differences per referent.
    :param df_summary: The dataframe containing the final information per referent.
    :param target_language: The target language.
    :return: `None`
    """
    match_conditions = [
        ("fem_embedding", "feminine"),
        ("masc_embedding", "masculine"),
        ("neut_embedding", "neutral"),
    ]
    mismatch_conditions = [
        ("fem_embedding", "masculine"),
        ("fem_embedding", "neutral"),
        ("masc_embedding", "feminine"),
        ("masc_embedding", "neutral"),
        ("neut_embedding", "masculine"),
        ("neut_embedding", "feminine"),
    ]

    df_match = df_summary[df_summary.apply(
        lambda row: (row["referent_embedding"], row["majority_gender"]) in match_conditions, axis=1
    )]
    df_mismatch = df_summary[df_summary.apply(
        lambda row: (row["referent_embedding"], row["majority_gender"]) in mismatch_conditions, axis=1
    )]

    for df, title, filename in [
        # Matches
        (df_match, f"{target_language} — Matched Embedding & Gender",
         os.path.join("output_v1", f"Referent_Dotplot_{target_language}_match.png")),
         # Mismatches
        (df_mismatch, f"{target_language} — Mismatched Embedding & Gender",
         os.path.join("output_v1", f"Referent_Dotplot_{target_language}_mismatch.png"))
    ]:
        if df.empty:
            print(f"No data for: {title}.")
            return

        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.18)))

        gender_colours = {
            "masculine": "#4C72B0",
            "feminine": "#DD8452",
            "neutral": "#696969"
        }

        embedding_markers = {
            embedding: marker for embedding, marker in zip(
                sorted(df["referent_embedding"].unique()),
                ["o", "s", "^"]
            )
        }

        df_sorted = df.sort_values("mean_probability", ascending=True).reset_index(drop=True)

        for embedding, group in df_sorted.groupby("referent_embedding"):

            for gender, subgroup in group.groupby("majority_gender"):
                y_positions = [df_sorted.index[df_sorted["referent"] == r].tolist()[0] for r in subgroup["referent"]]
                ax.scatter(
                    subgroup["mean_probability"],
                    y_positions,
                    label=f"{embedding} | {gender}",
                    marker=embedding_markers[embedding],
                    color=gender_colours.get(gender, "grey"),
                    s=60,
                    alpha=0.85,
                    zorder=3
                )
                ax.errorbar(
                    subgroup["mean_probability"],
                    y_positions,
                    xerr=subgroup["std_probability"],
                    fmt="none",
                    color=gender_colours.get(gender, "grey"),
                    alpha=0.4,
                    capsize=3,
                    zorder=2
                )

        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted["referent"], fontsize=8)
        ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_title(title, fontsize=20, fontweight="medium")
        ax.set_xlabel("Mean Probability Difference", fontsize=16)
        ax.set_ylabel("Referent", fontsize=16)
        ax.legend(title="Embedding | Majority Gender", loc="upper left", fontsize=12)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}.")

    # Print summary statistics
    for label, df in [("Match", df_match), ("Mismatch", df_mismatch)]:
        print(f"\n--- {target_language} {label} Statistics ---")
        print(f"Number of referents: {len(df)}")
        print(f"Overall average probability: {df['mean_probability'].mean():.4f}")
        print(f"Standard deviation: {df['mean_probability'].std():.4f}")


def plot_visualisations_prob_difference(d_prob_difference_dfs: Dict) -> None:
    """Create visualisations for overall probability differences.
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


def get_pos_pivot(df_summary: pd.DataFrame, keep_pos: List) -> Tuple[pd.DataFrame, pd.Series]:
    """Get pivot data for desired part-of-speech (POS) tags.
    :param df_summary: The dataframe containing the final information per referent.
    :param keep_pos: List of desired POS tags.
    :return: The pivot dataframe and the gender map dataframe.
    """
    rows = []

    for _, row in df_summary.iterrows():

        for pos in row["salient_pos"]:
            rows.append({
                "referent": row["referent"],
                "salient_pos": pos,
                "majority_gender": row["majority_gender"]
            })

    df = pd.DataFrame(rows)
    pivot = df.groupby(["referent", "salient_pos"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(columns=[p for p in keep_pos if p in pivot.columns], fill_value=0)

    # Build referent -> majority_gender mapping
    gender_map = df.drop_duplicates("referent").set_index("referent")["majority_gender"]

    return pivot, gender_map


def apply_gender_colours(ax: plt.Axes, gender_map: pd.Series, d_gender_colours: Dict) -> None:
    """Apply colour to gender axis ticks.
    :param ax: The axis.
    :param gender_map: The gender map dataframe.
    :param d_gender_colours: Dictionary containing the gender labels linked to their colours.
    :return: `None`
    """
    for tick in ax.get_yticklabels():
        referent = tick.get_text()
        gender = gender_map.get(referent, None)

        if gender:
            tick.set_color(d_gender_colours.get(gender, "black"))


def plot_visualisations_salient_words_combined(d_summary_salient_words_dfs: Dict) -> None:
    """Create visualisations for salient words per referent.
    :param d_prob_difference_dfs: Dictionary containing dataframes with probability differences per target language.
    :return: `None`
    """
    lang_1 = sorted(list(d_summary_salient_words_dfs.keys()), reverse=True)[0]
    lang_2 = sorted(list(d_summary_salient_words_dfs.keys()), reverse=True)[1]
    df_lang_1 = d_summary_salient_words_dfs[lang_1]
    df_lang_2 = d_summary_salient_words_dfs[lang_2]

    # ---- 1. Word Cloud (shared words only) ----
    all_words_1 = set([word for words in df_lang_1["salient_words"] for word in words])
    all_words_2 = set([word for words in df_lang_2["salient_words"] for word in words])
    shared_words = all_words_1 & all_words_2

    # Count frequencies of shared words across both languages combined
    all_words_combined = [word for words in df_lang_1["salient_words"] for word in words] + [word for words in df_lang_2["salient_words"] for word in words]
    word_freq_shared = Counter({word: count for word, count in Counter(all_words_combined).items() if word in shared_words})

    # Create word cloud
    wc = WordCloud(width=1200, height=600, background_color="white", colormap="coolwarm", max_words=100).generate_from_frequencies(word_freq_shared)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Shared Salient Words — {lang_1} & {lang_2}", fontsize=30, fontweight="medium")
    plt.tight_layout()
    plt.savefig(os.path.join("output_v1", f"Wordcloud_{lang_1}_{lang_2}_shared.png"), dpi=300, bbox_inches="tight")
    print(f"Saved: Wordcloud_{lang_1}_{lang_2}_shared.png")
    """plt.show()"""

    # ---- 3. Side-by-side Heatmaps: one per language ----
    keep_pos = ["NOUN", "VERB", "ADJ", "PROPN", "PRON", "ADV"]
    pivot_1, gender_map_1 = get_pos_pivot(df_lang_1, keep_pos)
    pivot_2, gender_map_2 = get_pos_pivot(df_lang_2, keep_pos)

    d_gender_colours = {"masculine": "#4C72B0", "feminine": "#DD8452", "neutral": "#696969"}
    all_referents = sorted(set(pivot_1.index) | set(pivot_2.index))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(10, len(all_referents) * 0.18)), sharey=False)

    sns.heatmap(pivot_1, cmap="YlOrRd", linewidths=0.3, linecolor="grey", ax=ax1, cbar_kws={"label": "Frequency"})
    ax1.set_title(f"{lang_1} — POS Frequency per Referent", fontsize=13, fontweight="medium")
    ax1.set_xlabel("POS Tag", fontsize=13)
    ax1.set_ylabel("Referent", fontsize=15)
    ax1.tick_params(axis="x", rotation=45, labelsize=10)
    ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=8, fontweight="bold")
    apply_gender_colours(ax1, gender_map_1, d_gender_colours)

    sns.heatmap(pivot_2, cmap="YlOrRd", linewidths=0.3, linecolor="grey", ax=ax2, cbar_kws={"label": "Frequency"})
    ax2.set_title(f"{lang_2} — POS Frequency per Referent", fontsize=13, fontweight="medium")
    ax2.set_xlabel("POS Tag", fontsize=13)
    ax2.set_ylabel("")
    ax2.tick_params(axis="x", rotation=45, labelsize=10)
    ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=8, fontweight="bold")
    apply_gender_colours(ax2, gender_map_2, d_gender_colours)

    # Add gender legend
    plt.tight_layout()
    plt.savefig(os.path.join("output_v1", f"POS_heatmap_{lang_1}_{lang_2}.png"), dpi=300, bbox_inches="tight")
    print(f"Saved: POS_heatmap_{lang_1}_{lang_2}.png")
    """plt.show()"""
