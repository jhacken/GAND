from .process_JSONs import load_json
import os
from pathlib import Path
import sys
from typing import Dict, List, Union
from xlsxwriter import Workbook


def filtered_sentences_to_xlsx(path_direc: Union[str, Path], referent: str, l_sents_filtered: List) -> None:
    """Write filtered sentences to XLSX file (one file per referent).
    :param path_direc: Path to directory in which spreadsheet need to be saved.
    :param referent: The referent.
    :param l_sents_filtered: List containing the filtered sentences for a given referent item.
    :return: `None`
    """
    if not os.path.isdir(path_direc):
        os.makedirs(path_direc)

    # Create spreadsheet
    fn_xlsx = f"{referent}.xlsx"
    path_xlsx = os.path.join(path_direc, fn_xlsx)
    wb = Workbook(path_xlsx)
    ws = wb.add_worksheet("selectedSents")

    # Define formatting
    cell_format_default = wb.add_format()
    cell_format_default.set_align("center")
    cell_format_default.set_align("vcenter")
    cell_format_wrap = wb.add_format()
    cell_format_wrap.set_align("vcenter")
    cell_format_wrap.set_text_wrap()
    bold = wb.add_format({"bold": True})
    bold.set_align("center")
    bold.set_align("vcenter")
    boldred = wb.add_format({"bold": True, "font_color": "red"})

    # Define column width
    ws.set_column(0, 0, 12.5, cell_format_default)
    ws.set_column(1, 2, 90, cell_format_wrap)

    # Define column names
    ws.write(0, 0, "checked", bold)
    ws.write(0, 1, "sentence_to_read", bold)
    ws.write(0, 2, "sentence_to_copy", bold)

    # Loop over sentences
    row = 1

    for tup_sent in l_sents_filtered:
        tup_idxs = tup_sent[0]
        l_toks = tup_sent[1]
        sentence_text = tup_sent[2]

        l_segments = []

        for idx, tok in enumerate(l_toks):

            if idx not in tup_idxs:
                l_segments.append(f"{tok} ")
            else:
                l_segments.append(boldred)
                l_segments.append(f"{tok} ")

        ws.write_rich_string(row, 1, *l_segments)
        ws.write(row, 2, sentence_text)
        row += 1
                
    wb.close()

    print(f"Spreadsheet with filtered sentences for '{referent}' saved into {path_direc}.\n\n")


def merge_data_per_item_into_single_xlsx(
        timestr: str, dataset_name: str, list_of_referents: List, d_referents_to_source: Dict, d_meta: Dict,
        path_direc_inp: Union[str, Path], path_direc_outp: Union[str, Path]
) -> None:
    """Join the selected sentences per item into one single overarching XLSX.
    :param timestr: Time string of current job.
    :param dataset_name: Name of the dataset to be filtered.
    :param list_of_referents: List containing the referent entities.
    :param d_referents_to_source: Dictionary in which the referent entities are linked to their source (i.e.
        the predefined TXT word list in which they are included).
    :param d_meta: Dictionary containing meta information about the filtering process.
    :param path_direc_inp: Path to directory in which JSONs containing sentences per referent are saved.
    :param path_direc_outp: Path to directory in which spreadsheet need to be saved.
    :return: `None`
    """
    if not os.path.isdir(path_direc_outp):
        os.makedirs(path_direc_outp)

    # Create spreadsheet
    fn_xlsx = f"sentences_selected_joined_{dataset_name}_{timestr}.xlsx"
    path_xlsx = os.path.join(path_direc_outp, fn_xlsx)
    wb = Workbook(path_xlsx)
    ws = wb.add_worksheet("selectedSents")

    # Define formatting
    cell_format_default = wb.add_format()
    cell_format_default.set_align("center")
    cell_format_default.set_align("vcenter")
    cell_format_wrap = wb.add_format()
    cell_format_wrap.set_align("vcenter")
    cell_format_wrap.set_text_wrap()
    bold = wb.add_format({"bold": True})
    bold.set_align("center")
    bold.set_align("vcenter")
    boldred = wb.add_format({"bold": True, "font_color": "red"})

    # Define column width
    ws.set_column(0, 0, 17.5, cell_format_default)
    ws.set_column(1, 1, 12.5, cell_format_default)
    ws.set_column(2, 3, 80, cell_format_wrap)
    ws.set_column(4, 4, 22.5, cell_format_default)

    # Define column names
    ws.write(0, 0, "referent", bold)
    ws.write(0, 1, "checked", bold)
    ws.write(0, 2, "sentence_to_read", bold)
    ws.write(0, 3, "sentence_to_copy", bold)
    ws.write(0, 4, "source", bold)

    # Loop over temporary JSONs to get sentences and write them to the spreadsheet
    n_sentences_final_dataset = 0
    d_freq_per_referent = {referent: 0 for referent in sorted(set(list_of_referents))}
    row = 1

    for doc in os.listdir(path_direc_inp):
        referent = doc.replace(".json", "")
        l_sents_filtered = load_json(os.path.join(path_direc_inp, doc))
        
        for l_sent in l_sents_filtered:
            n_sentences_final_dataset += 1
            d_freq_per_referent[referent] += 1
            tup_idxs = l_sent[0]
            l_toks = l_sent[1]
            sentence_text = l_sent[2]

            l_segments = []

            for idx, tok in enumerate(l_toks):

                if idx not in tup_idxs:
                    l_segments.append(f"{tok} ")
                else:
                    l_segments.append(boldred)
                    l_segments.append(f"{tok} ")

            ws.write(row, 0, referent)
            ws.write_rich_string(row, 2, *l_segments)
            ws.write(row, 3, sentence_text)
            ws.write(row, 4, d_referents_to_source[referent])
            row += 1

    # Set the autofilter.
    ws.autofilter(f"A1:E{n_sentences_final_dataset}")

    # Add metadata
    ws = wb.add_worksheet("meta")
    ws.set_column(0, 0, 50, cell_format_default)
    ws.set_column(1, 1, 25, cell_format_default)
    row = 0

    for criterion in d_meta:
        ws.write(row, 0, criterion)
        ws.write(row, 1, d_meta[criterion])
        row += 1

    
    wb.close()

    print(f"\t- Spreadsheet with selected sentences being joined saved into {path_direc_outp}.")
    print(f"\t- Number of filtered sentences after second processing step (total): {n_sentences_final_dataset}.")
    print(f"\t- Number of filtered sentences after second processing step (per referent):\n\t{d_freq_per_referent}.")


def sentences_filtered_out_to_txt(
        path_direc: Union[str, Path], dict_sents_filtered_out: Dict
) -> None:
    """Write sentences that were filtered out to TXT files (one file per filtering rule).
    :param path_direc: Path to directory in which TXTs need to be saved.
    :param dict_sents_filtered_out: Dictionary containing the sentences filtered out.
    :return: `None`
    """
    if not os.path.isdir(path_direc):
        os.makedirs(path_direc)

    # Loop iteratively over rules, referent entities, and sentences in dictionary
    for rule in dict_sents_filtered_out:

        with open(os.path.join(path_direc, f"{rule}.txt"), mode="w", encoding="utf-8") as f:

            for referent in dict_sents_filtered_out[rule]:
            
                for sentence in dict_sents_filtered_out[rule][referent]:
                    f.write(f"{referent}\t{sentence}\n")

        f.close()

    print(f"\t- TXTs with sentences filtered out saved into {path_direc} (one file per filtering rule).")
