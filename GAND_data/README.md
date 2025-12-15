# GAND (data)

In `GAND_CT_sample.tsv` (a version in XLSX format is also provided) you can find a sample of 100 instances of the GAND dataset, containing the following information:
- **referent**: ambiguous word
- **EN_source sentence**: original English sentence featuring the ambiguous word
- **referent_embedding**: list from which referent was taken (see `GAND_compilation` for more information)
- **sentence source**: data source from which the original English sentence was taken (see `GAND_compilation` for more information)
- **OPUS_DE_translation**: translation of source sentence into German using Opus-MT
- **DE_gender**: gender of the referent in the German translation
- **OPUS_ES_translation**: translation of source sentence into Spanish using Opus-MT
- **ES_gender**: gender of the referent in the Spanish translation
- **OPUS_DE_Contrastive**: manually contrasted version of the German translation in terms of gender
- **DE_contrastive_gender**: gender of the referent in the German contrastive translation
- **OPUS_ES_Contrastive**: manually contrasted version of the Spanish translation in terms of gender
- **ES_contrastive_gender**: gender of the referent in the Spanish contrastive translation

The full GAND dataset of 5000+ instances (and 500 for GAND-CT) will be made available upon acceptance of the paper.
