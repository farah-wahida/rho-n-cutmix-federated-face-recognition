# rho-n CutMix for Federated Face Recognition

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTBIOM.2026.3725635-blue)](https://doi.org/10.1109/TBIOM.2026.3725635)
[![Tests](https://github.com/farah-wahida/rho-n-cutmix-federated-face-recognition/actions/workflows/tests.yml/badge.svg)](https://github.com/farah-wahida/rho-n-cutmix-federated-face-recognition/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official publication code for **“An Input-Level Privacy Pipeline for Federated Face Recognition via rho-n CutMix and Keyed Block Permutation,”** IEEE Transactions on Biometrics, Behavior, and Identity Science, DOI [10.1109/TBIOM.2026.3725635](https://doi.org/10.1109/TBIOM.2026.3725635).

## Release contents

This publication release provides:

- multi-source rho-n CutMix with mixture labels;
- deterministic keyed 8 × 8 block permutation and authorized inverse decoding;
- ResNet-18 and ResNet-34 recognition models;
- five-client non-IID FedAvg training with the published optimization controls;
- the exact 4-dataset × 2-backbone × 5-setting configuration matrix, including run-specific confidence penalties;
- recognition, calibration, NLL, membership inference, biometric verification, adaptive inversion, identity correspondence, and latency evaluation;
- machine-readable reference values for paper Tables IV–X;
- an output-free quickstart notebook, tests, and continuous integration.

The original monolithic experiment notebook has been replaced by reusable package modules and command-line workflows. Exploratory cells, machine-specific paths, and embedded outputs are not part of the release.

## Scope and artifact availability

| Artifact | Included | Location or reason |
|---|---|---|
| Source code | Yes | src/privacy_pipeline/ and scripts/ |
| Exact experiment matrix | Yes | configs/paper_grid.yaml |
| Published table values | Yes | results/paper/ |
| Tutorial notebook | Yes | notebooks/publication_quickstart.ipynb |
| Datasets | No | Original licenses and privacy conditions apply |
| Trained checkpoints | No | Large research artifacts; rerun training |
| Prepared/reconstructed images | No | Derived biometric data are not redistributed |

Accordingly, the repository supports reference-result inspection and protocol reproduction. Exact numerical reruns additionally require the same licensed data release, file-level splits, software stack, and hardware behavior described in [the reproducibility guide](docs/reproducibility.md).

## Repository structure

~~~text
.
├── configs/
│   ├── base.yaml
│   └── paper_grid.yaml
├── docs/
│   ├── datasets.md
│   └── reproducibility.md
├── notebooks/
│   └── publication_quickstart.ipynb
├── results/paper/
│   └── table_iv_...csv through table_x_...csv
├── scripts/
│   ├── generate_configs.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── evaluate_membership.py
│   ├── evaluate_verification.py
│   ├── evaluate_inversion.py
│   ├── evaluate_latency.py
│   ├── reproduce_tables.py
│   └── validate_release.py
├── src/privacy_pipeline/
└── tests/
~~~

## Installation

Python 3.10 or later is supported.

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

## Validate the publication package

~~~bash
python scripts/validate_release.py
~~~

This verifies generation of all 40 concrete run configurations and the presence of all seven machine-readable reference tables.

## Inspect the published results

Open [the quickstart notebook](notebooks/publication_quickstart.ipynb), or load the CSV files directly:

~~~python
import pandas as pd

accuracy = pd.read_csv("results/paper/table_iv_accuracy.csv")
inversion = pd.read_csv("results/paper/table_vii_inversion.csv")
~~~

## Generate the exact run configurations

~~~bash
python scripts/generate_configs.py
~~~

The command writes 40 YAML files under configs/generated/. Settings are Raw and Defenses A–D:

| Setting | rho | Donor patches | Keyed permutation | Inverse decoder |
|---|---:|---:|---|---|
| Raw | — | — | No | No |
| Defense A | 0.8 | 2 | Yes | Yes |
| Defense B | 0.6 | 2 | Yes | Yes |
| Defense C | 0.8 | 3 | Yes | Yes |
| Defense D | 0.8 | 4 | Yes | Yes |

## Run one experiment

After arranging a licensed dataset according to [the data guide](docs/datasets.md):

~~~bash
python scripts/prepare_data.py --config configs/generated/pubfig/defense_a_resnet18.yaml

python scripts/train.py \
  --config configs/generated/pubfig/defense_a_resnet18.yaml \
  --validation-root data/prepared/pubfig/validation/defense_a

python scripts/evaluate.py \
  --config configs/generated/pubfig/defense_a_resnet18.yaml \
  --data-root data/prepared/pubfig/test/defense_a \
  --validation-root data/prepared/pubfig/validation/defense_a
~~~

Additional evaluations:

~~~bash
python scripts/evaluate_membership.py --config CONFIG --member-root TRAIN --nonmember-root TEST
python scripts/evaluate_verification.py --config CONFIG --data-root TEST
python scripts/evaluate_inversion.py --config CONFIG
python scripts/evaluate_latency.py --config CONFIG --data-root TEST
~~~

Each command writes machine-readable artifacts below the run output directory. See [the reproducibility guide](docs/reproducibility.md) for the full protocol and interpretation cautions.

## Extending the work

The components are deliberately separated:

- add an input transform beside RhoNCutMix in transforms.py;
- add a backbone in build_backbone in models.py;
- add an aggregation rule beside fedavg in federated.py;
- add attacks or metrics without changing the training pipeline;
- extend paper_grid.yaml and regenerate concrete configurations.

## Reproduction-key notice

Keys in example configurations are deterministic research reproduction keys, not operational secrets. Replace them with a securely supplied, untracked key in any deployment.

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
