# rho-n CutMix for Federated Face Recognition

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTBIOM.2026.3725635-blue)](https://doi.org/10.1109/TBIOM.2026.3725635)
[![Tests](https://github.com/farah-wahida/rho-n-cutmix-federated-face-recognition/actions/workflows/tests.yml/badge.svg)](https://github.com/farah-wahida/rho-n-cutmix-federated-face-recognition/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Clean research code for **“An Input-Level Privacy Pipeline for Federated Face Recognition via rho-n CutMix and Keyed Block Permutation,”** accepted in IEEE Transactions on Biometrics, Behavior, and Identity Science.

The implementation reorganizes the original experiment notebook into a reusable Python package. Repeated dataset/model cells, machine-specific paths, embedded outputs, and exploratory fragments have been removed. Data, trained weights, and generated attack images are not included.

## What the repository provides

- Multi-source rho-n CutMix with mixture labels
- Deterministic keyed block permutation and authorized inverse decoding
- ResNet-18 and ResNet-34 recognition models
- Non-IID client partitioning and FedAvg training
- Recognition accuracy, ECE, NLL, and validation-fitted temperature scaling
- Score-based membership inference and effective AUC
- Pairwise biometric verification metrics
- Sinkhorn-relaxed adaptive full-access inversion utilities
- A generated 40-run experiment matrix covering the paper settings
- Unit tests and continuous integration

## Pipeline

On each authorized client, donor patches are inserted into a base image, the composite is scrambled with a key-derived block permutation, and the tuple consisting of the scrambled image, soft label, and retained permutation index is used locally. A parameter-free decoder restores the authorized block order before recognition. Only model updates and sample counts participate in FedAvg; images, labels, permutation indices, and the key remain local.

## Repository structure

~~~text
.
├── configs/
│   ├── base.yaml
│   └── paper_grid.yaml
├── docs/
│   ├── datasets.md
│   └── reproducibility.md
├── scripts/
│   ├── generate_configs.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_membership.py
│   └── reproduce_tables.py
├── src/privacy_pipeline/
│   ├── attacks.py
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── federated.py
│   ├── models.py
│   └── transforms.py
└── tests/
~~~

## Installation

Python 3.10 or later is recommended.

~~~bash
git clone https://github.com/farah-wahida/rho-n-cutmix-federated-face-recognition.git
cd rho-n-cutmix-federated-face-recognition
python -m venv .venv
pip install -e .
~~~

For development:

~~~bash
pip install -r requirements-dev.txt
pytest -q
~~~

## Data

Arrange each training, validation, and test split in class folders. The repository uses the same ImageFolder interface for PubFig, LFW, PINS, and CIFAR-10.

~~~text
data/raw/pubfig/train/<identity>/*.jpg
data/raw/pubfig/validation/<identity>/*.jpg
data/raw/pubfig/test/<identity>/*.jpg
~~~

See [dataset preparation](docs/datasets.md) for the full layout, class counts, and defended-split workflow.

## Generate the paper configurations

~~~bash
python scripts/generate_configs.py
~~~

This creates 40 YAML files under configs/generated: four datasets, two backbones, and Raw plus Defenses A-D.

| Setting | rho | Donor patches | Keyed permutation | Inverse decoder |
|---|---:|---:|---|---|
| Raw | — | — | No | No |
| Defense A | 0.8 | 2 | Yes | Yes |
| Defense B | 0.6 | 2 | Yes | Yes |
| Defense C | 0.8 | 3 | Yes | Yes |
| Defense D | 0.8 | 4 | Yes | Yes |

## Run an experiment

Prepare a defended training split:

~~~bash
python scripts/prepare_data.py \
  --config configs/generated/pubfig/defense_a_resnet18.yaml
~~~

Train with FedAvg:

~~~bash
python scripts/train.py \
  --config configs/generated/pubfig/defense_a_resnet18.yaml
~~~

Evaluate on a held-out test split while fitting temperature only on validation data:

~~~bash
python scripts/evaluate.py \
  --config configs/generated/pubfig/defense_a_resnet18.yaml \
  --data-root data/prepared/pubfig/test/defense_a \
  --validation-root data/prepared/pubfig/validation/defense_a
~~~

Evaluate score-based membership inference:

~~~bash
python scripts/evaluate_membership.py \
  --config configs/generated/pubfig/defense_a_resnet18.yaml \
  --member-root data/prepared/pubfig/train/defense_a \
  --nonmember-root data/prepared/pubfig/test/defense_a
~~~

Aggregate machine-readable run outputs:

~~~bash
python scripts/reproduce_tables.py
~~~

The full sequencing, result scope, and conditions needed for numerical reproduction are described in [the reproducibility guide](docs/reproducibility.md).

## Extending the work

The package components are intentionally independent:

- Add a transformation beside RhoNCutMix in transforms.py.
- Add a backbone in build_backbone in models.py.
- Add an aggregation rule beside fedavg in federated.py.
- Add an attack or privacy metric without changing training code.
- Add dataset and defense definitions to paper_grid.yaml and regenerate configurations.

The adaptive inversion routine accepts a logits function, so alternative model interfaces and normalization schemes can be evaluated without coupling attack optimization to one checkpoint format.

## Reproduction-key notice

Keys committed in example YAML files are deterministic reproduction keys, not secrets. Replace them with an untracked, securely supplied key for any operational deployment.

## Citation

~~~bibtex
@article{wahida2026input,
  author  = {Farah Wahida and Ibrahim Khalil},
  title   = {An Input-Level Privacy Pipeline for Federated Face Recognition via rho-n CutMix and Keyed Block Permutation},
  journal = {IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year    = {2026},
  doi     = {10.1109/TBIOM.2026.3725635}
}
~~~

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## Acknowledgement

This work was supported by the RMIT Research Stipend Scholarship (RRSS).

## License

The code is released under the [MIT License](LICENSE). Dataset licenses and usage conditions remain with their respective owners.
