from datasets import IterableDataset
from sentence_splitter import SentenceSplitter
from spacy.lang.en import English
import sys
from tqdm import tqdm
from typing import Dict, List


def filter_referent_entities(
        dataset_name: str, list_of_referents: List, dataset: IterableDataset, len_dataset: int, n_texts_to_analyse: int,
        min_sentence_length: int, max_sentence_length: int
) -> Dict:
    """Filter the dataset by excluding all sentences that do not include any of the referent entities.
    :param dataset_name: Name of the dataset to be filtered.
    :param list_of_referents: List containing the referent entities.
    :param dataset: The dataset.
    :param len_dataset: The number of texts in the dataset.
    :param n_texts_to_analyse: The number of texts from the dataset to analyse.
    :param min_sentence_length: The minimum number of tokens a sentence must have.
    :param max_sentence_length: The maximum number of tokens a sentence is allowed to have.
    :return: A dictionary with the referent entities as the keys and lists of retrieved sentences as the values.
    """
    # Load sentence splitter and tokeniser
    splitter = SentenceSplitter(language="en")
    nlp_spacy_for_tokenizer = English()
    spacy_tokenizer = nlp_spacy_for_tokenizer.tokenizer

    # Loop over randomly shuffled dataset (only consider number of items defined in `n_texts_to_analyse`)
    print(f"\t- Looping over {n_texts_to_analyse:,d} of the {len_dataset:,d} texts in the dataset "
          f"(= {int(n_texts_to_analyse/len_dataset*100)}%) to identify sentences containing the referent entities ...")
    dataset_shuffled = dataset.shuffle(seed=42) if dataset_name in ["C4", "Wikipedia"] else dataset
    set_of_referents = set(list_of_referents)
    dict_sentences_per_referent = {referent: [] for referent in sorted(set_of_referents)}
    n_matches = 0
    
    for idx, item in tqdm(enumerate(dataset_shuffled), total=n_texts_to_analyse):

        if idx >= n_texts_to_analyse:
            break

        text = item["text"]
        list_sents = splitter.split(text=text)

        for sent in list_sents:
            words = [tok.text for tok in spacy_tokenizer(sent)]

            # Sentences that do not meet sentence length criteria are discarded
            if min_sentence_length <= len(words) <= max_sentence_length:
                    
                # Sentences in which first character is not a capital letter are discarded, as are sentences including 
                # a (semi-)colon
                if words[0].isalpha() and words[0][0].isupper() and not any(item in words for item in [";", ":"]):
                    matches = set_of_referents.intersection(set(words))

                    for match in matches:
                        n_matches += 1
                        dict_sentences_per_referent[match].append(sent)

    print(f"\t- Number of filtered sentences after first processing step (total): {n_matches}.")
    d_freq_per_referent_entity = {referent: len(dict_sentences_per_referent[referent]) for referent in sorted(set_of_referents)}
    print(f"\t- Number of filtered sentences after first processing step (per referent entity):\n\t{d_freq_per_referent_entity}.")

    return dict_sentences_per_referent
