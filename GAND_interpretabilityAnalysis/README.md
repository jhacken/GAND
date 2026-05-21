# GAND (interpretability analysis)

Run the commands below to replicate the interpretability analysis presented in Hackenbuchner et al. (forthcoming). Make sure to run the commands from this directory as your working directory. Output files are saved into an automatically created output folder.

1. Compute contrastive translation attribution

```shell
$ python compute_CT_attribution_v1.py DE,ES
```

2. Analyse attribution and probability

```shell
$ python analyse_attribution_and_probability_v1.py DE,ES
```

**NOTE**: This script relies on the output files from the script that computes the contrastive translation attribution, so that script needs to be executed first.

3. Analyse POS and dependency distance

```shell
$ python analyse_POS_and_dependencyDistance_v1.py DE,ES
```

**NOTE**: This script relies on the output files from the script that computes the contrastive translation attribution, so that script needs to be executed first.

## Requirements

### Python

See `requirements.txt`. Tested with Python 3.13.5 and CUDA 13.0. The requirements can be installed automatically in one go by running the following command in your shell (preferably in a fresh virtual environment):

```shell
$ python -m pip install -r requirements.txt
```
