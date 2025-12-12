import inseq
import pandas as pd
import spacy
from spacy.tokens import Doc, Token
import string
import sys
from typing import Dict, List, Optional


def find_base_word_fallback(doc: Doc, base_word: str) -> Optional[Token]:
    """Fallback function to find just the base word if contraction sequence fails.
    :param doc: The spaCy Doc.
    :param base_word: The base word.
    :return: The identified token.
    """
    print(f"\tDEBUG: Falling back to finding just base word '{base_word}'")

    for token in doc:

        if token.text.lower() == base_word.lower():
            print(f"\tDEBUG: Found base word fallback: '{token.text}' (POS: {token.pos_})")
            return token

    print(f"\tDEBUG: Base word fallback failed for '{base_word}'")

    return None


def find_word_apostrophe_sequence(doc: Doc, base_word: str, apostrophe_part: str) -> Optional[Token]:
    """Find a sequence like ['word', \"'s\"] or ['I', \"'m\"].
    :param doc: The spaCy Doc.
    :param base_word: The base word.
    :param apostrophe_part: The part of the word starting at the apostrophe.
    :return: The identified token.
    """
    # Debug: print what we're looking for and what's available
    print(f"\tDEBUG: Looking for base_word='{base_word}' + apostrophe_part='{apostrophe_part}'")

    for i in range(len(doc) - 1):
        current_token = doc[i].text.lower()
        next_token = doc[i + 1].text
        print(f"\tDEBUG: Checking tokens[{i}]='{doc[i].text}' (lower: '{current_token}') + tokens[{i+1}]='{doc[i + 1].text}' (repr: {repr(next_token)})")

        # Check if current token matches base word
        if current_token == base_word.lower():
            print(f"\tDEBUG: Base word '{base_word}' matches token '{doc[i].text}'")

            # Now check apostrophe part - be very flexible with matching
            if (next_token == apostrophe_part or
                next_token.lower() == apostrophe_part.lower() or
                next_token == "'s" or  # regular apostrophe
                next_token == "'s" or  # smart quote
                next_token == "′s" or  # prime symbol
                next_token == "`s" or  # backtick
                (apostrophe_part.lower() == "'s" and next_token in ["'s", "'s", "′s", "`s"]) or
                (apostrophe_part.lower() == "'m" and next_token in ["'m", "'m", "′m", "`m"]) or
                (apostrophe_part.lower() == "'re" and next_token in ["'re", "'re", "′re", "`re"]) or
                (apostrophe_part.lower() == "'ll" and next_token in ["'ll", "'ll", "′ll", "`ll"]) or
                (apostrophe_part.lower() == "'ve" and next_token in ["'ve", "'ve", "′ve", "`ve"]) or
                (apostrophe_part.lower() == "'d" and next_token in ["'d", "'d", "′d", "`d"])):
                print(f"\tDEBUG: Found match! Base='{doc[i].text}' + Apostrophe='{next_token}'")
                return doc[i]
            else:
                print(f"\tDEBUG: Base word matches but apostrophe part '{next_token}' doesn't match expected '{apostrophe_part}'")

    print(f"\tDEBUG: No match found for '{base_word}' + '{apostrophe_part}'")

    return None


def find_hyphenated_sequence(doc: Doc, hyphenated_word: str) -> Optional[Token]:
    """Find a sequence of tokens that make up a hyphenated word.
    :param doc: The spaCy Doc.
    :param hyphenated_word: The hyphenated word.
    :return: The identified token.
    """
    parts = hyphenated_word.split("-")

    for i in range(len(doc) - (len(parts) * 2 - 1) + 1):
        match = True
        token_idx = i

        for j, part in enumerate(parts):

            if token_idx >= len(doc) or doc[token_idx].text.lower() != part.lower():
                match = False
                break

            token_idx += 1

            if j < len(parts) - 1:

                if token_idx >= len(doc) or doc[token_idx].text != "-":
                    match = False
                    break

                token_idx += 1

        if match:
            return doc[i]

    return None


def find_contraction_sequence(doc, contraction_word) -> Optional[Token]:
    """Find a sequence of tokens that make up a contraction or possessive.
    :param doc: The spaCy Doc.
    :param contraction_word: The contraction or possessive word.
    :return: The identified token.
    """
    contraction_lower = contraction_word.lower()

    if contraction_lower.endswith("'s"):
        base_word = contraction_lower[:-2]
        result = find_word_apostrophe_sequence(doc, base_word, "'s")

        if result:
            return result

        # Fallback: just find the base word
        return find_base_word_fallback(doc, base_word)

    elif contraction_lower.endswith("'m"):
        base_word = contraction_lower[:-2]
        result = find_word_apostrophe_sequence(doc, base_word, "'m")

        if result:
            return result

        return find_base_word_fallback(doc, base_word)

    elif contraction_lower.endswith("'re"):
        base_word = contraction_lower[:-3]
        result = find_word_apostrophe_sequence(doc, base_word, "'re")

        if result:
            return result

        return find_base_word_fallback(doc, base_word)

    elif contraction_lower.endswith("'ll"):
        base_word = contraction_lower[:-3]
        result = find_word_apostrophe_sequence(doc, base_word, "'ll")

        if result:
            return result

        return find_base_word_fallback(doc, base_word)

    elif contraction_lower.endswith("'ve"):
        base_word = contraction_lower[:-3]
        result = find_word_apostrophe_sequence(doc, base_word, "'ve")

        if result:
            return result

        return find_base_word_fallback(doc, base_word)

    elif contraction_lower.endswith("'d"):
        base_word = contraction_lower[:-2]
        result = find_word_apostrophe_sequence(doc, base_word, "'d")

        if result:
            return result

        return find_base_word_fallback(doc, base_word)

    else:
        return None


def find_salient_token(doc: Doc, salient_word: str) -> Optional[Token]:
    """Find salient word token, handling punctuation, contractions, and hyphenated words.
    :param doc: The spaCy Doc.
    :param salient_word: The salient word that needs to be identified in the sentence.
    :return: The identified token.
    """
    salient_lower = salient_word.lower().strip()

    # Try exact match first
    for token in doc:

        if token.text.lower() == salient_lower:
            return token

    # Try matching with lemma (base form)
    for token in doc:

        if token.lemma_.lower() == salient_lower:
            return token

    # Handle hyphenated words by finding token sequences
    if "-" in salient_lower:
        hyphen_result = find_hyphenated_sequence(doc, salient_lower)

        if hyphen_result:
            return hyphen_result

    # Handle contractions and possessives by finding token sequences
    if "'" in salient_lower:
        contraction_result = find_contraction_sequence(doc, salient_lower)

        if contraction_result:
            return contraction_result

    # More comprehensive matching for other cases
    for token in doc:
        token_text = token.text.lower()

        # 1. Clean token by removing trailing punctuation
        token_clean = token_text.rstrip('.,!?;:"\'()[]{}')

        if token_clean == salient_lower:
            return token

        # 2. Handle single letters (like 'C')
        if len(salient_lower) == 1:

            if token_clean == salient_lower:
                return token

            # For single letters, also check if token starts with the letter
            if token_text.startswith(salient_lower) and len(token_clean) == 1:
                return token

            # Special case: look for "D.C" when looking for "C"
            if salient_lower == "c" and ("d.c" in token_text or "dc" in token_clean):
                return token

        # 3. Try matching by removing all punctuation from both
        salient_no_punct = salient_lower.translate(str.maketrans("", "", string.punctuation))
        token_no_punct = token_text.translate(str.maketrans("", "", string.punctuation))

        if salient_no_punct and token_no_punct and salient_no_punct == token_no_punct:
            return token

    return None


def save_all_normalized_merged_source_attributions_for_first_arrow_word(
        out: inseq.FeatureAttributionOutput, data: pd.DataFrame, sentence_index: int, nlp_spacy: spacy.Language
) -> Optional[Dict]:
    """Prints the top merged source word attributions for the first target token containing '→', after normalizing
    source attributions for that target token so that they sum to 1. Subword tokens are merged into whole words before
    ranking and printing. Also prints the indices of the constituent source tokens. Removes the specific target word
    for this sentence based on sentence_index.
    :param out: Output from the attribution model.
    :param data: GAND data.
    :param sentence_index: Index of the sentence in the input file.
    :param nlp_spacy: spaCy model.
    :return: A dictionary combining all the processed information.
    """
    # Create dictionaries to later fill with contrastive probability distribution, source tokens and attributions, and
    # POS tags, and combine the two dicts into one at the end of the function
    prob_dict = {}
    source_tokens_dict = {}

    # Get source tokens and optionally their ids and original index
    out_data = out.sequence_attributions[0]
    source_tokens = [t.token if hasattr(t, "token") else str(t) for t in out_data.source]
    target_tokens = [t.token if hasattr(t, "token") else str(t) for t in out_data.target]

    """print(f"\n--- SENTENCE {sentence_index + 1} ---")
    print(source_tokens)
    print(target_tokens)"""

    # Remove end of line character from source tokens
    """print("Removing </s> from source tokens.")"""

    if "</s>" in source_tokens:
        source_tokens.remove("</s>")

    # Handle alignment window (important for correct attribution indexing)
    attr_pos_start = getattr(out_data, "attr_pos_start", 0)
    attr_pos_end = getattr(out_data, "attr_pos_end", len(target_tokens))
    aligned_target_tokens = target_tokens[attr_pos_start:attr_pos_end]  # Only tokens with attributions

    # Prepare attribution matrix (handle possible 3D tensor)
    saliency_heatmap = out_data.source_attributions

    if hasattr(saliency_heatmap, "shape") and len(saliency_heatmap.shape) == 3:
        saliency_heatmap = saliency_heatmap.sum(dim=2)

    contrast_scores = out_data.step_scores["contrast_prob_diff"]

    # Find first target token containing "→" in the aligned window
    target_word_index = next((i for i, token in enumerate(aligned_target_tokens) if "→" in token), None)

    if target_word_index is None:
        """print("No target token containing '→' found.")"""
        return None

    tgt_token = aligned_target_tokens[target_word_index]
    contrast_val = float(contrast_scores[target_word_index])

    # Collect and normalize attribution scores for this target token (column in matrix)
    raw_scores = [
        float(
            saliency_heatmap[i][target_word_index].item()
            if hasattr(saliency_heatmap[i][target_word_index], "item") else saliency_heatmap[i][target_word_index]
        ) for i in range(len(source_tokens))
    ]
    score_sum = sum(raw_scores)
    normalized_scores = [s / score_sum if score_sum != 0 else 0.0 for s in raw_scores]

    # Merge subwords into words and sum their attribution scores
    merged_words = []
    current_word = ""
    current_score = 0.0
    current_indices = []
    current_pos = ""

    for idx, (token, score) in enumerate(zip(source_tokens, normalized_scores)):

        if token.startswith("▁") or token in [".", "</s>"]:  # Start of a new word or special

            if current_word:
                merged_words.append((current_word, current_score, current_indices, current_pos))

            # Start new word
            current_word = token.lstrip("▁")
            current_score = score
            current_indices = [idx]

        else:
            current_word += token
            current_score += score
            current_indices.append(idx)

    # Add last word
    if current_word:
        merged_words.append((current_word, current_score, current_indices, current_pos))

    # Get the specific row for this sentence_index
    row = data.iloc[sentence_index]
    text = row["EN_source_sentence"]
    referent = row["referent"]

    # Get the specific target word for this sentence
    current_target_word = referent.lower()
    """print(f"Referent word for sentence {sentence_index + 1}: '{current_target_word}'")"""

    # Create a new list excluding the specific target word for this sentence
    filtered_words = []

    # Extract individual words from compound target words
    all_target_words = set()

    if current_target_word:
        all_target_words.add(current_target_word)  # Add the full compound

        # Add individual words from compounds
        for word_part in current_target_word.split():
            all_target_words.add(word_part)

    """print(f"All referent words to remove (including compound parts): {all_target_words}")"""

    # Parse the text with spaCy once
    doc = nlp_spacy(text)

    for word, score, indices, pos in merged_words:

        # Check if the word itself is in target_words (including compound parts)
        if word.lower().strip(".,:\"'—.-?![]()") in all_target_words:
            """print(f"Removing '{word}' - found as a referent")"""
            continue

        # Check if any individual word from a multi-word phrase is in target_words
        word_parts = word.lower().split()

        if any(part in all_target_words for part in word_parts):
            """print(f"Removing '{word}' - contains part of the referent")"""
            continue

        # For each salient word, find it in the text and get its POS
        salient_token = find_salient_token(doc, word)

        if salient_token:
            chosen_pos = salient_token.pos_
            """print(f"\tFound '{word}' with POS tag: {chosen_pos}")"""
            filtered_words.append((word, score, indices, chosen_pos))
        else:

            # Try matching without punctuation as a fallback
            found = False

            for token in doc:

                if not token.is_space and token.text == word.strip(".,:\"'—.-?![]()"):
                    chosen_pos = token.pos_
                    """print(f"\tFound '{word}' (cleaned) with POS tag: {chosen_pos}")"""
                    filtered_words.append((word, score, indices, chosen_pos))
                    found = True
                    break

            if not found:
                """print(f"Salient word '{word}' not found in text: {text[:50]}...")"""

                # Enhanced debugging for contractions - check the WORD, not salient_token
                if "'" in word:
                    print(f"\tDEBUG: Apostrophe word - analyzing tokenization...")
                    tokens_with_indices = [(i, token.text, repr(token.text)) for i, token in enumerate(doc)]
                    print(f"\tDEBUG: All tokens with repr: {tokens_with_indices}")

                    # Show character codes for apostrophe characters
                    for char in word:
                        if char in "''′`":
                            print(f"\tDEBUG: Apostrophe character '{char}' has Unicode code: {ord(char)}")

                """# Show available tokens
                tokens_in_text = [token.text for token in doc]
                print(f"\tAvailable tokens: {tokens_in_text}")
                similar_tokens = [
                    token.text for token in doc
                    if word.lower().replace("'", "").replace("'", "") in token.text.lower()
                       or token.text.lower() in word.lower().replace("'", "").replace("'", "")
                ]

                if similar_tokens:
                    print(f"\tSimilar tokens found: {similar_tokens}")"""

    merged_words = filtered_words

    # Sort merged words by total attribution score, descending
    merged_words.sort(key=lambda x: x[1], reverse=True)

    """print(f"\nFirst target word containing '→' (index {target_word_index + attr_pos_start}): '{tgt_token}'")
    print(f"Contrastive prob diff: {contrast_val:.6f}")"""

    """# Print all merged attributions in order for inspection
    print("All merged source words and their normalised attributions:")
    
    for word, score, indices, pos in merged_words:
        print(f"\tSource word '{word}' [{pos}] (source token idx {indices}): normalised attribution score = {score:.6f}")"""

    # Fill dictionaries with contrastive probability difference, and with source tokens and attribution scores and POS
    # tags
    prob_dict[tgt_token] = contrast_val

    for word, score, indices, pos in merged_words:
        source_tokens_dict[word] = [score, pos]

    combined_dict = {"prob_data": prob_dict, "token_data": source_tokens_dict}

    """print(f"Source tokens dict: {source_tokens_dict}")
    print(f"Prob dict: {prob_dict}")
    print(f"Combined dict: {combined_dict}")"""

    return combined_dict


def get_sums_scores_per_sentence(all_contrastive_attributions: Dict, stopwords: List, percent: float) -> Dict:
    """Process the attributions dictionary including: contr. prob. difference, source tokens, salient scores and POS
    tags. Removes stopwords for this sentence. Takes top 20% of source tokens based on the sum of attribution scores
    PER SENTENCE.
    :param all_contrastive_attributions: Attribution dictionary.
    :param stopwords: Stopwords list.
    :param percent: Top x% of salient words chosen based on total attribution value per length.
    :return: Dictionary with salient words.
    """
    salient_dictionary = {}
    punctuation_marks = {".", ",", ":", '"', "'", "—", "-", "?", "!", "[", "]", "(", ")"}

    for index, individual_sentence_dict in all_contrastive_attributions.items():

        # Initialize empty dictionaries for this index
        prob_dict = {}
        salient_tokens_dict = {}

        # Skip if the dictionary is empty
        if not individual_sentence_dict:
            print(f"\n=== Skipping Sentence Index {index} (empty dictionary) ===")

            # Add empty combined_dict for this index
            salient_dictionary[index] = {"prob_data": {}, "salient_data": {}}
            continue

        # Also skip if token_data is empty
        if "token_data" not in individual_sentence_dict or not individual_sentence_dict["token_data"]:
            print(f"\n=== Skipping Sentence Index {index} (no token_data) ===")

            # Add empty combined_dict for this index
            salient_dictionary[index] = {"prob_data": {}, "salient_data": {}}
            continue

        """print(f"\n=== Processing Sentence Index {index} ===")"""

        # Get data from the dictionary
        attr_words = list(individual_sentence_dict["token_data"].keys())
        attr_scores = list(attribution_score[0] for attribution_score in individual_sentence_dict["token_data"].values())
        attr_pos = list(attribution_score[1] for attribution_score in individual_sentence_dict["token_data"].values())

        # Filter out stopwords and punctuation marks while keeping words, scores, and POS aligned
        filtered_data = []

        for word, score, pos in zip(attr_words, attr_scores, attr_pos):

            # Skip if the word itself is just punctuation
            if word in punctuation_marks:
                """print(f"\tSkipping punctuation: '{word}'")"""
                continue

            # Skip if the cleaned word is a stopword
            cleaned_word = word.lower().strip(".,:\"'—.-?![]()")

            if cleaned_word in stopwords:
                """print(f"\tSkipping stopword: '{word}'")"""
                continue

            # Keep this word-score-pos triplet
            filtered_data.append((word, score, pos))

        # Unzip the filtered data back into separate lists
        if filtered_data:
            attr_words, attr_scores, attr_pos = zip(*filtered_data)
            attr_words = list(attr_words)
            attr_scores = list(attr_scores)
            attr_pos = list(attr_pos)
        else:
            attr_words = []
            attr_scores = []
            attr_pos = []

        attr_difference_words = [contr_difference for contr_difference in individual_sentence_dict["prob_data"].keys()]
        attr_difference = [contr_difference for contr_difference in individual_sentence_dict["prob_data"].values()]
        """print(f"Contrastive difference probability for {attr_difference_words[0]}: {attr_difference[0]}")"""

        # Create a list of tuples (word, score, pos) and sort by score
        word_score_pairs = list(zip(attr_words, attr_scores, attr_pos))
        """print(f"Word score pairs: {word_score_pairs}")"""

        # Sort by score in descending order to get highest scores first
        word_score_pairs.sort(key=lambda x: x[1], reverse=True)
        selected_words = []
        total_score_per_sentence = 0.0

        for word, score, pos in word_score_pairs:
            """print(f"\tSource word '{word}' [{pos}]: normalised attribution score = {score:.6f}")"""
            total_score_per_sentence += score

        """print(f"\nTOTAL SCORE FOR {index}: {total_score_per_sentence:.6f}")"""

        top_x_percent = total_score_per_sentence * percent
        """print(f"TOP {percent*100}% OF {index}: {top_x_percent:.6f}")"""

        intermediate_score = 0.0

        for word, score, pos in word_score_pairs:

            if intermediate_score < top_x_percent:
                intermediate_score += score
                selected_words.append((word, score, pos))
            else:
                break

        """top_n = len(selected_words)
        print(f"\nFor top {percent*100}%, selected {top_n} words with summed score of {intermediate_score:.6f}:")

        for word, score, pos in selected_words:
            print(f"\tSource word '{word}' [{pos}]: normalised attribution score = {score:.6f}")

        top_pairs = word_score_pairs[:top_n]"""

        # Fill dictionaries with contrastive probability difference, and with source tokens and attribution scores and
        # POS tags
        prob_dict[attr_difference_words[0]] = attr_difference[0]

        for word, score, pos in selected_words:
            salient_tokens_dict[word] = [score, pos]

        salient_dictionary[index] = {"prob_data": prob_dict, "salient_data": salient_tokens_dict}

    """print(f"Salient dict: {salient_dictionary}")"""

    return salient_dictionary
