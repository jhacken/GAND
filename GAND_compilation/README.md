# GAND (compilation)

## Data sources

To build GAND, data from two different sources were used: C4 (Common Crawl) and Open Subtitles.

- C4: We used the `en` subset from the `allenai/c4` dataset, which is available on [Hugging Face](https://huggingface.co/datasets/allenai/c4). C4 contains 364,868,892 texts (with 456 tokens per text on average in the first 1,000 texts of the randomly shuffled dataset).
- Open Subtitles: We used the monolingual English data from the [Open Subtitles corpus](https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles), compiled within the [OPUS project](https://opus.nlpl.eu/). The Open Subtitles corpus contains 2,739,528 texts (in this case, a text corresponds to a single subtitle; 8 tokens per subtitle on average in the first 1,000 subtitles of the randomly shuffled dataset).

The C4 dataset is available through Hugging Face and does not need to be downloaded locally in order to be accessed. The Open Subtitles corpus, however, does need to be downloaded beforehand. To this end, please execute the command provided below in your shell (from the `GAND_compilation` directory as your working directory):

```shell
# Linux
$ wget -P input_v1/downloadedDataSources/OpenSubtitles https://object.pouta.csc.fi/OPUS-OpenSubtitles/v1/mono/en.txt.gz
$ gunzip input_v1/downloadedDataSources/OpenSubtitles/en.txt.gz
```

## Data processing

The data processing script (called `process_dataSources_v1.py`) performs five main steps, which are discussed in greater detail below. To run the script yourself, you can simply execute the following command in your shell. In the default setting, only the name of the dataset ("C4" or "OpenSubtitles") needs to be specified. In the command below, the Open Subtitles dataset will be loaded and processed.

```shell
$ python process_dataSources_v1.py OpenSubtitles
```

For more information on the optional parameters that can be specified (i.e. the device on which the script should be run, the number of texts to be analysed, minimum and maximum sentence length, and the maximum number of sentences to be selected per referent entity), have a look at the [source code](https://github.com/jhacken/GAND/blob/bdda2a89d3590ea56169b01403845cc8d419da76/GAND_compilation/process_dataSources_v1.py#L78) or run the command below.

```shell
$ python process_dataSources_v1.py -h
```

To create the pool of candidate sentences from which the final GAND dataset was created, the following parameters were used:

```shell
$ python process_dataSources_v1.py C4 --device cuda:0 --n_texts_to_analyse 250000 --min_sentence_length 5 --max_sentence_length 30 --max_n_sentences_per_referent 250

$ python process_dataSources_v1.py OpenSubtitles --device cuda:0 --n_texts_to_analyse 250000 --min_sentence_length 5 --max_sentence_length 30 --max_n_sentences_per_referent 250
```

### STEP_1

In the first step, five predefined lists of referent entities (i.e. gender-ambiguous English words) are loaded from the `input_v1/predefinedWordLists` directory (descriptions of the lists are included in the GAND paper):

- `female_embedding_list.txt`
- `male_embedding_list.txt`
- `LLM_female_list.txt`
- `LLM_male_list.txt`
- `LLM_neutral_list.txt`

This part of the script uses the function(s) included in `utils_v1/process_resources.py`.

### STEP_2

During the second step, the chosen dataset is loaded. This part of the script makes use of the function(s) included in `utils_v1/load_rawDataSources.py`.

### STEP_3

The third step of the script iterates over the texts in the (randomly shuffled) dataset and extracts all sentences that (1) contain at least one of the referent entities, (2) start with an uppercased letter, (3) do not contain a (semi-)colon, and (4) meet the minimum and maximum sentence length criteria. This part of the script makes use of the function(s) included in `utils_v1/filter_data_referentEntities.py`.

### STEP_4

In the fourth step, a set of handcrafted rules is applied to exclude sentences that (1) are unsuitable (e.g., no conjugated verb or wrong part of speech) or that (2) might include a (co-)reference that provides clues that allow the reader to disambiguate the person word. Regarding co-references, we check for references to proper nouns or to "gender (pro)nouns". The list of "gender (pro)nouns" is manually compiled and consists of the following sets of words:

- he, she, him, her, his, hers, himself, herself
- mother, father, mom, dad, sister, brother, wife, husband, grandmother, grandfather, daughter, son
- lady, sir, woman, man, women, men, female, male, girl, boy

This part of the script uses the function(s) included in `utils_v1/filter_data_coreference.py`. To perform the morphosyntactic analysis of the sentences, we make use of [Stanza's NLP toolkit](https://stanfordnlp.github.io/stanza/). To install the Stanza model of your choice, please refer to the [Stanza documentation](https://stanfordnlp.github.io/stanza/download_models.html). Detailed information on the handcrafted rules (including examples) is included in the GAND paper.

### STEP_5

In the fifth and final step, the sentences that passed all the filters are joined into one single XLSX file, which is automatically saved into the `output_v1/sentences_selected_joined` directory. Additionally, the sentences that were filtered out are written to TXT files and saved into the `output_v1/sentences_filtered_out` directory (one file per rule). This part of the script uses the function(s) included in `utils_v1/write_output.py`.

## Creation of final GAND dataset

[to describe]

## Requirements

### Python

See `requirements.txt`. Tested with Python 3.13.5. The requirements can be installed automatically in one go by running the following command in your shell (preferably in a fresh virtual environment):

```shell
$ python -m pip install -r requirements.txt
```
