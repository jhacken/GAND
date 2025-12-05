import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Union


def load_list_of_referents(path_direc: Union[str, Path]) -> Tuple[List, Dict]:
    """Load predefined lists of referent entities and join them into one single list.
    :param path_direc: Path to directory in which predefined lists are saved.
    :return: The joined list of referent entities and a dictionary in which the words are linked to their source.
    """
    list_of_referents = []
    d_referents_to_source = {}

    for doc in sorted(os.listdir(path_direc)):
        source = doc.replace(".txt", "")

        with open(os.path.join(path_direc, doc), "r") as reader:
            list_of_referents_doc = [referent.strip("\n") for referent in reader.readlines()]
        reader.close()

        list_of_referents += list_of_referents_doc

        for referent in list_of_referents_doc:
            
            if referent in d_referents_to_source:
                print(f"\n--- WARNING ----\n"
                      f"When processing '{doc}', it appeared that one of the words had been processed before: '{referent}'.\n"
                      f"\t- Entry that will be kept: {(referent, d_referents_to_source[referent])}.\n"
                      f"\t- Entry that will be skipped: {(referent, source)}.\n")
            else:
                d_referents_to_source[referent] = source

    print(f"\t- The list of referents contains {len(list_of_referents)} items.\n"
          f"\t- The first ten items from the (unsorted) list: {list_of_referents[:10]}.")

    return list_of_referents, d_referents_to_source
