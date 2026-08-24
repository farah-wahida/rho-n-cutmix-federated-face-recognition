# Reproducibility guide

## Experiment matrix

configs/paper_grid.yaml describes the complete 4-dataset × 2-backbone × 5-setting matrix. Generate the 40 concrete YAML files with:

~~~bash
python scripts/generate_configs.py
~~~

The transformation settings are:

| Setting | rho | Donor patches | Keyed permutation | Inverse decoder |
|---|---:|---:|---|---|
| Raw | — | — | No | No |
| Defense A | 0.8 | 2 | Yes | Yes |
| Defense B | 0.6 | 2 | Yes | Yes |
| Defense C | 0.8 | 3 | Yes | Yes |
| Defense D | 0.8 | 4 | Yes | Yes |

The grid uses five clients, three local epochs, Dirichlet alpha 0.5, an 8 × 8 block grid, and a 64-permutation codebook. Change a generated YAML file when reproducing a run whose archived metadata records a different confidence-penalty coefficient or optimizer setting.

## Recommended run order

1. Fix and record the dataset splits.
2. Generate the 40 experiment configurations.
3. Prepare training, validation, and test data for each defended setting.
4. Train the raw and defended models.
5. Evaluate recognition utility and validation-fitted calibration.
6. Evaluate member/non-member score attacks.
7. Run adaptive inversion for the selected targets and aggregate the outputs.

Use a separate output directory for every seed. The scripts write checkpoints and machine-readable JSON rather than embedding results in notebooks.

## Result aggregation

~~~bash
python scripts/reproduce_tables.py \
  --outputs outputs \
  --name evaluation.json \
  --output results/evaluation_summary.csv
~~~

Repeat with --name membership.json for membership-inference results.

## Scope of this release

This repository is a clean, reusable implementation extracted from the research notebook. Dataset files, trained weights, intermediate images, and notebook outputs are intentionally excluded. Exact numerical agreement additionally requires the same data releases, identity filters, splits, preprocessing, software versions, random seeds, and archived run-specific settings used for the paper.
