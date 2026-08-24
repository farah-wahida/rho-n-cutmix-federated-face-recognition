# Dataset preparation

The repository does not redistribute PubFig, LFW, PINS, or CIFAR-10. Obtain each dataset under its original terms and create fixed training, validation, and test splits before running an experiment.

## Expected layout

Every split uses the ImageFolder convention:

~~~text
data/raw/
├── pubfig/
│   ├── train/<identity>/*.jpg
│   ├── validation/<identity>/*.jpg
│   └── test/<identity>/*.jpg
├── lfw/
├── pins/
└── cifar10/
~~~

For CIFAR-10, export the images into class folders first. The loader deliberately uses the same folder interface for all four datasets.

The paper evaluates 62 PubFig identities, 62 LFW identities, 105 PINS identities, and all 10 CIFAR-10 classes. Keep the class-to-index ordering identical across the three splits.

## Defended splits

Defended experiments read a generated manifest.csv and its associated images. Generate each split separately with the same defense settings and reproduction key. For example, start from the generated Defense A configuration, then change only dataset.root and dataset.prepared_root:

~~~text
data/prepared/pubfig/
├── train/defense_a/
├── validation/defense_a/
└── test/defense_a/
~~~

Run:

~~~bash
python scripts/prepare_data.py --config path/to/split_config.yaml
~~~

A prepared manifest records the scrambled image path, mixture label, retained permutation index, and donor indices. Raw images, the reproduction key, and permutation information are not transmitted in the federated protocol.

## Private deployment keys

The YAML files contain deterministic reproduction keys so independent runs can be compared. They are not operational secrets. Use an untracked key supplied by a secret manager in any real deployment.
