from .process_JSONs import dump_json
from .write_output import filtered_sentences_to_xlsx
import os
import stanza
import sys
from tqdm import tqdm
from typing import Dict, Optional


def filter_coreference(
        timestr: str, dataset_name: str, dict_sentences_per_referent: Dict, device: str,
        max_n_sentences_per_referent: Optional[int],
        *,
        apply_filter_conjugated_verb: bool = True, exclude_sentences_with_multiple_instances_referent_word: bool = True,
        write_to_xlsx: bool = False
) -> Dict:
    """Further filter the dictionary of sentences collected in STEP_3 by excluding sentences showing co-reference.
    Sentences are processed using Stanza (https://stanfordnlp.github.io/stanza/data_objects#document).
    The filtered set of sentences is saved in an XLSX file per referent.
    :param timestr: Time string of current job.
    :param dataset_name: Name of the dataset to be filtered.
    :param dict_sentences_per_referent: Dictionary containing the previously collected sentences.
    :param device: Device on which the script should be run.
    :param max_n_sentences_per_referent: The maximum number of sentences that should be selected for a given referent
        entity. As soon as this threshold is reached, the script will go on to the next word.
    :param apply_filter_conjugated_verb: Indicates whether the filter that eliminates sentences without a conjugated
        verb should be activated or not. Defaults to `True`.
    :param exclude_sentences_with_multiple_instances_referent_word: Indicates whether sentences in which the referent
        occurs more than once should be excluded or not. Defaults to `True`.
    :param write_to_xlsx: Indicates whether results per item should be written to separate XLSXs. Defaults to `False`.
    :return: A dictionary containing the sentences that were filtered out.
    """
    if max_n_sentences_per_referent is None:
        max_n_sentences_per_referent_orig = None
        max_n_sentences_per_referent = max([
            len(dict_sentences_per_referent[ref]) for ref in dict_sentences_per_referent
        ])

    # Load Stanza model
    if device == "cpu":
        nlp = stanza.Pipeline(
            "en", processors="tokenize,mwt,pos,lemma,depparse", use_gpu=False, tokenize_no_ssplit=True
        )
    else:
        nlp = stanza.Pipeline(
            "en", processors="tokenize,mwt,pos,lemma,depparse", device=device, tokenize_no_ssplit=True
        )

    # Sentences that were filtered out will be stored here
    l_rules = [
        "CONJ-VERB",
        "OTHER-POS", "COMPOUND", "GENDER-PROPN-CHILDREN", "GENDER-PROPN-CHILDREN-HEAD",
        "GENDER-PROPN-OBLIQUE", "GENDER-PROPN-NMOD-APPOS", "GENDER-PROPN-XCOMP",
        "GENDER-PROPN-ROOT", "GENDER-PROPN-NSUBJ"
    ]
    dict_sents_filtered_out = {
        f"RULE_{rule_name}": {item: [] for item in dict_sentences_per_referent}
        for rule_name in l_rules
    }

    # Define gender-specific (pro)nouns
    set_gender_refs = {
        "he", "she", "him", "her", "his", "hers", "himself", "herself",
        "mother", "father", "mom", "dad", "sister", "brother", "wife", "husband", "grandmother", "grandfather",
        "daughter", "son",
        "lady", "sir", "woman", "man", "women", "men", "female", "male", "girl", "boy"
    }

    # Loop over dictionary
    d_stats = {
        "n_unique_sentences_post": int, "max_n_sentences_per_referent": max_n_sentences_per_referent_orig,
        "d_freq_per_referent_entity": {}
    }
    l_sents_in_all = []

    for referent, list_of_sentences in tqdm(list(dict_sentences_per_referent.items())):
        print(f"Processing {len(list_of_sentences)} sentences for '{referent}' ...")

        n_sents_processed = 0
        max_n_sentences_per_referent_reached = False
        l_sents_in = []  # Empty list in which the sentences that pass all the filters will be stored
        
        l_docs_nlp = nlp.bulk_process(list_of_sentences) # Call the neural pipeline on this list of documents

        for doc in l_docs_nlp:
            n_sents_processed += 1
            sentence = doc.text

            l_toks_obj = []
            l_toks_text = []
            l_pos = []
            l_feats = []
            l_heads = []
            l_deprels = []
            l_deprels_full = []

            for idx, tok in enumerate(doc.iter_words()):
                l_toks_obj.append(tok)
                l_toks_text.append(tok.text)
                l_pos.append(tok.upos)
                l_feats.append(tok.feats if tok.feats is not None else "")
                l_heads.append(tok.head - 1 if tok.head != 0 else idx)
                l_deprels.append(tok.deprel.split(":")[0])
                l_deprels_full.append(tok.deprel)
                
            # APPLY RULES
            include_sentence = True
            referent_identified_as_separate_token = False
            n_occurrences_referent_word = 0

            # ----- RULES AT SENTENCE LEVEL -----

            # RULE_CONJ-VERB
            if apply_filter_conjugated_verb:
                l_check_conjugated_verb = [True if "VerbForm=Fin" in feats else False for feats in l_feats]

                if True not in set(l_check_conjugated_verb):
                    include_sentence = False
                    dict_sents_filtered_out["RULE_CONJ-VERB"][referent].append(sentence)
                    continue

            # ----- RULES AT TOKEN LEVEL -----
            for idx_tok, tok in enumerate(l_toks_obj):
                
                if tok.text == referent:
                    referent_identified_as_separate_token = True
                    n_occurrences_referent_word += 1

                    # indices of head and head of head (index of token itelf is already defined in `idx_tok`)
                    idx_head = tok.head - 1 if tok.head != 0 else idx_tok
                    idx_head_of_head = l_heads[idx_head]

                    # information for children of referent word
                    d_children_referent_idx_deprel = {
                        idx: l_deprels[idx] for idx, head in enumerate(l_heads) if head == idx_tok and idx != idx_tok
                    }

                    l_toks_children_referent = [l_toks_text[idx] for idx, head in enumerate(l_heads) if head == idx_tok]

                    # information for children of head
                    d_children_head_idx_pos = {
                        idx: l_pos[idx] for idx, head in enumerate(l_heads) if head == idx_head and idx != idx_head
                    }
                    d_children_head_idx_deprel = {
                        idx: l_deprels[idx] for idx, head in enumerate(l_heads) 
                        if head == idx_head and idx != idx_head
                    }

                    l_toks_children_head = [l_toks_text[idx] for idx, head in enumerate(l_heads) if head == idx_head]
                    l_pos_children_head = [l_pos[idx] for idx, head in enumerate(l_heads) if head == idx_head]

                    l_children_head_tup_deprel_plus_pos = [
                        (l_deprels[idx], l_pos[idx]) for idx, head in enumerate(l_heads) 
                        if head == idx_head and idx != idx_head
                    ]
                    
                    # information for children of head of head
                    d_children_head_of_head_idx_deprel = {
                        idx: l_deprels[idx] for idx, head in enumerate(l_heads) 
                        if head == idx_head_of_head and idx != idx_head_of_head
                    }

                    l_toks_children_head_of_head = [
                        l_toks_text[idx] for idx, head in enumerate(l_heads) if head == idx_head_of_head
                    ]
                    l_pos_children_head_of_head = [
                        l_pos[idx] for idx, head in enumerate(l_heads) if head == idx_head_of_head
                    ]

                    l_children_head_of_head_tup_tok_plus_deprel = [
                        (l_toks_text[idx], l_deprels[idx]) for idx, head in enumerate(l_heads) 
                        if head == idx_head_of_head and idx != idx_head_of_head
                    ]
                    l_children_head_of_head_tup_deprel_plus_pos = [
                        (l_deprels[idx], l_pos[idx]) for idx, head in enumerate(l_heads) 
                        if head == idx_head_of_head and idx != idx_head_of_head
                    ]
                    l_children_head_of_head_tup_tok_plus_pos = [
                        (l_toks_text[idx], l_pos[idx]) for idx, head in enumerate(l_heads) 
                        if head == idx_head_of_head and idx != idx_head_of_head
                    ]
                    
                    # RULE_OTHER-POS
                    if tok.upos not in ["NOUN", "PROPN"]:
                        include_sentence = False
                        dict_sents_filtered_out["RULE_OTHER-POS"][referent].append(sentence)
                        break

                    # RULE_COMPOUND
                    if l_deprels[idx_tok] in ["compound", "amod"]:  # "amod" added to capture parsing errors as in "adult children"
                        include_sentence = False
                        dict_sents_filtered_out["RULE_COMPOUND"][referent].append(sentence)
                        break

                    if l_deprels[idx_tok] == "conj" and l_deprels[idx_head] in ["amod", "compound"]:
                        include_sentence = False
                        dict_sents_filtered_out["RULE_COMPOUND"][referent].append(sentence)
                        break

                    # RULE_GENDER-PROPN-CHILDREN
                    l_target_deprels = ["nmod", "appos", "flat", "vocative", "conj"]  # "conj" added because parsed like this in phrases such as "human rights activist and laywer Nasrin Sotoudeh"
                    l_pos_children_referent_custom = [
                        l_pos[idx] for idx, head in enumerate(l_heads) 
                        if head == idx_tok and l_deprels[idx] in l_target_deprels
                    ]

                    # Circumvent tagging errors with modifier being labelled as second subject (e.g., "In 2020 
                    # governor Andy Beshear appointed Barber to be circuit judge [...]") or as second object
                    if l_deprels[idx_tok] == "nsubj":
                        l_target_deprels += ["nsubj"]

                    if l_deprels[idx_tok] == "obj":
                        l_target_deprels += ["obj"]

                    l_pos_children_head_custom = [
                        l_pos[idx] for idx, head in enumerate(l_heads) 
                        if head == idx_head and l_deprels[idx] in l_target_deprels
                    ]

                    # Apply rule
                    if (any(item.lower() in set_gender_refs for item in l_toks_children_referent)
                            or "PROPN" in l_pos_children_referent_custom):
                        include_sentence = False
                        dict_sents_filtered_out["RULE_GENDER-PROPN-CHILDREN"][referent].append(sentence)
                        break

                    if l_deprels[idx_tok] in ["conj", "nsubj", "obj"]:
                        
                        if (any(item.lower() in set_gender_refs for item in l_toks_children_head)  
                                or "PROPN" in l_pos_children_head_custom):
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-CHILDREN"][referent].append(sentence)
                            break

                    break_loop = False
                    
                    for idx_child in [
                        idx for idx in d_children_referent_idx_deprel if d_children_referent_idx_deprel[idx] == "conj"
                    ]:
                        l_toks_children_child = [
                            l_toks_text[idx] for idx, head in enumerate(l_heads) if head == idx_child
                        ]
                        l_pos_children_child = [
                            l_pos[idx] for idx, head in enumerate(l_heads) if head == idx_child
                        ]

                        if (any(item.lower() in set_gender_refs for item in l_toks_children_child) 
                                    or "PROPN" in l_pos_children_child):
                            include_sentence = False
                            break_loop = True

                    if break_loop:
                        dict_sents_filtered_out["RULE_GENDER-PROPN-CHILDREN"][referent].append(sentence)
                        break

                    # RULE_GENDER-PROPN-CHILDREN-HEAD
                    if l_pos[idx_head] == "VERB" and "VERB" in d_children_head_idx_pos.values():
                        break_loop = False

                        for idx_child in [
                            idx for idx in d_children_head_idx_pos if d_children_head_idx_pos[idx] == "VERB"
                        ]:
                            l_toks_children_child = [
                                l_toks_text[idx] for idx, head in enumerate(l_heads) 
                                if head == idx_child and l_deprels[idx] == "nsubj"
                            ]
                            l_pos_children_child = [
                                l_pos[idx] for idx, head in enumerate(l_heads) 
                                if head == idx_child and l_deprels[idx] == "nsubj"
                            ]
                            
                            if (any(item.lower() in set_gender_refs for item in l_toks_children_child) 
                                    or "PROPN" in l_pos_children_child):
                                include_sentence = False
                                break_loop = True
                        
                        if break_loop:
                            dict_sents_filtered_out["RULE_GENDER-PROPN-CHILDREN-HEAD"][referent].append(sentence)
                            break

                    # RULE_GENDER-PROPN-OBLIQUE
                    if l_deprels[idx_tok] == "obl":
                        
                        if any(item.lower() in set_gender_refs for item in l_toks_children_head):
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-OBLIQUE"][referent].append(sentence)
                            break

                        if ("nsubj", "PROPN") in l_children_head_tup_deprel_plus_pos:
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-OBLIQUE"][referent].append(sentence)
                            break

                        if (("cc", "CCONJ") in l_children_head_tup_deprel_plus_pos 
                                and ("nsubj", "PROPN") in l_children_head_of_head_tup_deprel_plus_pos):
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-OBLIQUE"][referent].append(sentence)
                            break

                        if (("cc", "CCONJ") in l_children_head_tup_deprel_plus_pos 
                                and set([(tup[0].lower(), tup[1]) for tup in l_children_head_of_head_tup_tok_plus_deprel]).intersection(set([(ref, "nsubj") for ref in set_gender_refs]))):
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-OBLIQUE"][referent].append(sentence)
                            break

                        if ("obj", "PROPN") in l_children_head_tup_deprel_plus_pos:
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-OBLIQUE"][referent].append(sentence)
                            break

                        if l_deprels[idx_head] == "conj":

                            if (any(item.lower() in set_gender_refs for item in l_toks_children_head_of_head) 
                                    or ("nsubj", "PRON") in l_children_head_of_head_tup_deprel_plus_pos):
                                include_sentence = False
                                dict_sents_filtered_out["RULE_GENDER-PROPN-OBLIQUE"][referent].append(sentence)
                                break

                    if l_deprels[idx_tok] == "conj" and l_deprels[idx_head] == "obl":
                        
                        if any(item.lower() in set_gender_refs for item in l_toks_children_head_of_head):
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-OBLIQUE"][referent].append(sentence)
                            break

                        if ("nsubj", "PROPN") in l_children_head_of_head_tup_deprel_plus_pos:
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-OBLIQUE"][referent].append(sentence)
                            break

                        if ("obj", "PROPN") in l_children_head_of_head_tup_deprel_plus_pos:
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-OBLIQUE"][referent].append(sentence)
                            break

                    # RULE_GENDER-PROPN-NMOD-APPOS
                    if l_deprels[idx_tok] in ["nmod", "appos"]:
                            
                        if any(item.lower() in set_gender_refs for item in l_toks_children_head):
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-NMOD-APPOS"][referent].append(sentence)
                            break

                        if l_pos[idx_head] == "PROPN":
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-NMOD-APPOS"][referent].append(sentence)
                            break

                        if (l_pos[idx_head_of_head] == "VERB" 
                                and any((item.lower(), "nsubj") in l_children_head_of_head_tup_tok_plus_deprel for item in set_gender_refs)):
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-NMOD-APPOS"][referent].append(sentence)
                            break

                        if (l_pos[idx_head_of_head] == "VERB" 
                                and ("PROPN", "nsubj") in l_children_head_of_head_tup_tok_plus_pos):
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-NMOD-APPOS"][referent].append(sentence)
                            break

                        if (l_deprels[idx_head] == "xcomp" 
                                and any((item.lower(), "nsubj") in l_children_head_of_head_tup_tok_plus_deprel for item in set_gender_refs)):
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-NMOD-APPOS"][referent].append(sentence)
                            break

                        if (l_deprels[idx_head] == "xcomp" 
                                and ("PROPN", "nsubj") in l_children_head_of_head_tup_tok_plus_pos):
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-NMOD-APPOS"][referent].append(sentence)
                            break

                    if l_deprels[idx_tok] == "conj":
                            
                        if l_deprels[idx_head] in ["nmod", "appos"] and l_pos[idx_head_of_head] == "PROPN":
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-NMOD-APPOS"][referent].append(sentence)
                            break

                    # RULE_GENDER-PROPN-XCOMP
                    if l_deprels[idx_tok] == "xcomp":
                        break_loop = False
                            
                        if (any(item.lower() in set_gender_refs for item in l_toks_children_head) 
                                or "PROPN" in l_pos_children_head):
                            include_sentence = False
                            dict_sents_filtered_out["RULE_GENDER-PROPN-XCOMP"][referent].append(sentence)
                            break                            

                        for idx_child in [
                            idx for idx in d_children_head_idx_deprel 
                            if d_children_head_idx_deprel[idx] == "nsubj"
                        ]:
                            l_toks_children_child = [
                                l_toks_text[idx] for idx, head in enumerate(l_heads) if head == idx_child
                            ]
                            l_pos_children_child = [
                                l_pos[idx] for idx, head in enumerate(l_heads) if head == idx_child
                            ]
                            
                            if (any(item.lower() in set_gender_refs for item in l_toks_children_child) 
                                    or "PROPN" in l_pos_children_child):
                                include_sentence = False
                                break_loop = True

                        if l_deprels[idx_head] == "advcl":
                            break_loop = False

                            if (any(item.lower() in set_gender_refs for item in l_toks_children_head_of_head) 
                                    or "PROPN" in l_pos_children_head_of_head):
                                include_sentence = False
                                dict_sents_filtered_out["RULE_GENDER-PROPN-XCOMP"][referent].append(sentence)
                                break

                            for idx_child in [
                                idx for idx in d_children_head_of_head_idx_deprel 
                                if d_children_head_of_head_idx_deprel[idx] == "nsubj"
                            ]:
                                l_toks_children_child = [
                                    l_toks_text[idx] for idx, head in enumerate(l_heads) if head == idx_child
                                ]
                                l_pos_children_child = [
                                    l_pos[idx] for idx, head in enumerate(l_heads) if head == idx_child
                                ]

                                if (any(item.lower() in set_gender_refs for item in l_toks_children_child) 
                                        or "PROPN" in l_pos_children_child):
                                    include_sentence = False
                                    break_loop = True

                        if l_deprels[idx_head] == "conj":

                            if (any(item.lower() in set_gender_refs for item in l_toks_children_head_of_head) 
                                    or "PROPN" in l_pos_children_head_of_head):
                                include_sentence = False
                                dict_sents_filtered_out["RULE_GENDER-PROPN-XCOMP"][referent].append(sentence)
                                break

                            for idx_child in [
                                idx for idx in d_children_head_of_head_idx_deprel 
                                if d_children_head_of_head_idx_deprel[idx] == "nsubj"
                            ]:
                                l_toks_children_child = [
                                    l_toks_text[idx] for idx, head in enumerate(l_heads) if head == idx_child
                                ]
                                l_pos_children_child = [
                                    l_pos[idx] for idx, head in enumerate(l_heads) if head == idx_child
                                ]
                                
                                if (any(item.lower() in set_gender_refs for item in l_toks_children_child) 
                                        or "PROPN" in l_pos_children_child):
                                    include_sentence = False
                                    break_loop = True

                            if break_loop:
                                dict_sents_filtered_out["RULE_GENDER-PROPN-XCOMP"][referent].append(sentence)
                                break

                    # RULE_GENDER-PROPN-ROOT
                    if l_deprels[idx_tok] == "root" and "nsubj" in d_children_referent_idx_deprel.values():
                        break_loop = False

                        for idx_child in [
                            idx for idx in d_children_referent_idx_deprel if d_children_referent_idx_deprel[idx] == "nsubj"
                        ]:

                            if l_toks_text[idx_child] in set_gender_refs or l_pos[idx_child] == "PROPN":
                                include_sentence = False
                                break_loop = True

                        if break_loop:
                            dict_sents_filtered_out["RULE_GENDER-PROPN-ROOT"][referent].append(sentence)
                            break

                    if l_deprels[idx_tok] == "conj" and l_deprels[idx_head] == "root":
                        break_loop = False

                        for idx_child in [
                            idx for idx in d_children_head_idx_deprel 
                            if d_children_head_idx_deprel[idx] == "nsubj"
                        ]:

                            if l_toks_text[idx_child] in set_gender_refs or l_pos[idx_child] == "PROPN":
                                include_sentence = False
                                break_loop = True

                        # Circumvent tagging error where subject is wrongly linked to "conj" word (i.e. 
                        # the referent as head)
                        for idx_child in [
                            idx for idx in d_children_referent_idx_deprel if d_children_referent_idx_deprel[idx] == "nsubj"
                        ]:
                            if l_toks_text[idx_child] in set_gender_refs or l_pos[idx_child] == "PROPN":
                                include_sentence = False
                                break_loop = True

                        if break_loop:
                            dict_sents_filtered_out["RULE_GENDER-PROPN-ROOT"][referent].append(sentence)
                            break

                    # RULE_GENDER-PROPN-NSUBJ
                    if l_deprels[idx_tok] == "nsubj":

                        if l_pos[idx_head] != "VERB":

                            if l_toks_text[idx_head] in set_gender_refs or l_pos[idx_head] == "PROPN":
                                include_sentence = False
                                dict_sents_filtered_out["RULE_GENDER-PROPN-NSUBJ"][referent].append(sentence)
                                break

                        else:
                            break_loop = False

                            for idx_child in [
                                idx for idx in d_children_head_idx_deprel 
                                if d_children_head_idx_deprel[idx] == "xcomp"
                            ]:

                                if l_toks_text[idx_child] in set_gender_refs or l_pos[idx_child] == "PROPN":
                                    include_sentence = False
                                    break_loop = True

                            if break_loop:
                                dict_sents_filtered_out["RULE_GENDER-PROPN-NSUBJ"][referent].append(sentence)
                                break

            if exclude_sentences_with_multiple_instances_referent_word and n_occurrences_referent_word != 1:
                include_sentence = False

            # If sentence passes all filters, append it to the dedicated list
            if include_sentence and referent_identified_as_separate_token:
                tup_idxs_referent = tuple(
                    [idx for idx, tok in enumerate(l_toks_text) if tok == referent]
                )
                l_sents_in.append((tup_idxs_referent, l_toks_text, sentence))
                l_sents_in_all.append(sentence)

            # If maximum number of sentences is reached, write results to XLSX and go to next word
            if n_sents_processed != len(list_of_sentences):

                if len(l_sents_in) == max_n_sentences_per_referent:
                    max_n_sentences_per_referent_reached = True
                    print(f"Maximum number of {max_n_sentences_per_referent} filtered sentences is reached for "
                          f"'{referent}'.")
                    dump_json(
                        os.path.join("temp_v1", "sentences_selected_perItem", f"{dataset_name}_{timestr}"),
                        f"{referent}.json",
                        l_sents_in
                    )

                    if write_to_xlsx:
                        filtered_sentences_to_xlsx(
                            os.path.join("output_v1", "sentences_selected_perItem", f"{dataset_name}_{timestr}"),
                            referent, l_sents_in
                        )
                    
                    break

        # If maximum number is never reached, write results to XLSX after processing the final sentence
        if not max_n_sentences_per_referent_reached:
            dump_json(
                os.path.join("temp_v1", "sentences_selected_perItem", f"{dataset_name}_{timestr}"),
                f"{referent}.json",
                l_sents_in
            )
            d_stats["d_freq_per_referent_entity"][referent] = len(l_sents_in)

            if write_to_xlsx:
                filtered_sentences_to_xlsx(
                    os.path.join("output_v1", "sentences_selected_perItem", f"{dataset_name}_{timestr}"),
                    referent, l_sents_in
                )

    d_stats["n_unique_sentences_post"] = len(set(l_sents_in_all))
    dump_json(
        os.path.join("output_v1", "stats"),
        f"d_stats_filter_data_coreference_{dataset_name}_{timestr}.json",
        d_stats,
        indent=2
    )

    return dict_sents_filtered_out
