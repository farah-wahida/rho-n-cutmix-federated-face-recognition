# Reproducibility guide

## What is reproducible from this release

This release contains the reusable implementation, the exact 40-run configuration matrix, all evaluation entry points, and machine-readable reference values for Tables IV–X. It excludes datasets, checkpoints, prepared privacy images, and reconstructed samples because those artifacts are licensed, privacy-sensitive, or too large for Git.

Run the release audit first:

~~~bash
python scripts/validate_release.py
~~~

## Experiment matrix

Generate all 4-dataset × 2-backbone × 5-setting configurations:

~~~bash
python scripts/generate_configs.py
~~~

The generator applies the run-specific confidence-penalty coefficients recorded in configs/paper_grid.yaml; do not replace them with one global value. The common protocol uses five clients, three local epochs, Dirichlet alpha 0.5, Adam, cosine learning-rate decay, gradient clipping at 5.0, patience 8, an 8 × 8 block grid, and a 64-permutation codebook.

| Setting | rho | Donor patches | Keyed permutation | Inverse decoder |
|---|---:|---:|---|---|
| Raw | — | — | No | No |
| Defense A | 0.8 | 2 | Yes | Yes |
| Defense B | 0.6 | 2 | Yes | Yes |
| Defense C | 0.8 | 3 | Yes | Yes |
| Defense D | 0.8 | 4 | Yes | Yes |

## Dataset split contract

Each split must use class directories with stable identity names. Keep a manifest containing relative path, identity, split, and SHA-256 hash. PubFig, LFW, and PINS remain subject to their original licenses; CIFAR-10 can be downloaded through torchvision. See docs/datasets.md.

Numerical agreement requires the same source releases, filters, exact file-level splits, and library/hardware behavior. The repository therefore distinguishes:

- **reference reproduction**: inspect the committed paper tables;
- **protocol reproduction**: rerun the published code on a compatible data release;
- **exact numerical reproduction**: additionally requires the original non-redistributable data and checkpoints.

## Run order

~~~bash
python scripts/generate_configs.py
python scripts/prepare_data.py --config configs/generated/pubfig/defense_a_resnet18.yaml
python scripts/train.py --config configs/generated/pubfig/defense_a_resnet18.yaml --validation-root data/prepared/pubfig/validation/defense_a
python scripts/evaluate.py --config configs/generated/pubfig/defense_a_resnet18.yaml --data-root data/prepared/pubfig/test/defense_a --validation-root data/prepared/pubfig/validation/defense_a
python scripts/evaluate_membership.py --config configs/generated/pubfig/defense_a_resnet18.yaml --member-root data/prepared/pubfig/train/defense_a --nonmember-root data/prepared/pubfig/test/defense_a
python scripts/evaluate_verification.py --config configs/generated/pubfig/defense_a_resnet18.yaml --data-root data/prepared/pubfig/test/defense_a
python scripts/evaluate_latency.py --config configs/generated/pubfig/defense_a_resnet18.yaml --data-root data/prepared/pubfig/test/defense_a
python scripts/evaluate_inversion.py --config configs/generated/pubfig/defense_a_resnet18.yaml
~~~

Adaptive inversion uses ten target classes, three random restarts per target, 400 maximum iterations, Adam at 0.05, TV weight 5×10⁻⁴, L2 weight 2×10⁻³, permutation-entropy weight 0.05, and ten Sinkhorn iterations. The best restart is selected by target-class probability.

## Reference results and fresh outputs

Published values live in results/paper/. Fresh runs write checkpoint and JSON/CSV artifacts below outputs/<dataset>/<setting>/<backbone>/seed42/. Aggregate fresh results with:

~~~bash
python scripts/reproduce_tables.py --outputs outputs --name evaluation.json --output results/evaluation_summary.csv
~~~

Latency values in the paper are workstation measurements and must not be treated as mobile or embedded-device benchmarks.
