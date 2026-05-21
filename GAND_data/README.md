# GAND (data)

The full GAND dataset of 5000+ instances (split into a training, test, and validation/development set) is included in `GAND_full.zip` and also made available on Hugging Face: <https://huggingface.co/datasets/jhacken/GAND>.

In `GAND_CT_1000.tsv` (a version in XLSX format is also provided) you can find the 1,000 randomly sampled sentences from the GAND dataset for which contrastive translations were created, containing the following information:
- **ID_GAND_CT**: ID of the data instance in the contrastive translation sample
- **ID_GAND**: ID of the data instance in the full GAND dataset
- **referent**: ambiguous word
- **EN_source sentence**: original English sentence featuring the ambiguous word
- **referent_embedding**: list from which referent was taken (see `GAND_compilation` for more information)
- **sentence source**: data source from which the original English sentence was taken (see `GAND_compilation` for more information)
- **OPUS_DE_translation**: translation of source sentence into German using Opus-MT
- **DE_gender**: gender of the referent in the German translation
- **OPUS_ES_translation**: translation of source sentence into Spanish using Opus-MT
- **ES_gender**: gender of the referent in the Spanish translation
- **DE_Contrastive_TR**: manually contrasted version of the German translation in terms of gender
- **DE_contrastive_gender**: gender of the referent in the German contrastive translation
- **ES_Contrastive_TR**: manually contrasted version of the Spanish translation in terms of gender
- **ES_contrastive_gender**: gender of the referent in the Spanish contrastive translation
