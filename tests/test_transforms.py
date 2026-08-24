import torch
from torch.utils.data import Dataset

from privacy_pipeline.transforms import KeyedBlockPermutation, RhoNCutMix


class ToyDataset(Dataset):
    def __init__(self):
        self.targets = [0, 1, 2, 3]
        self.images = [
            torch.full((3, 16, 16), fill_value=index / 4)
            for index in range(4)
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


def test_keyed_permutation_round_trip():
    transform = KeyedBlockPermutation(
        grid_size=4,
        codebook_size=8,
        root_key="test_reproduction_key",
    )
    image = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(3, 16, 16)
    scrambled, permutation_index = transform.apply(image, sample_index=7)
    restored = transform.invert(scrambled, permutation_index)
    assert torch.equal(restored, image)


def test_keyed_permutation_is_deterministic():
    first = KeyedBlockPermutation(4, 8, "test_reproduction_key")
    second = KeyedBlockPermutation(4, 8, "test_reproduction_key")
    assert torch.equal(first.codebook, second.codebook)
    assert first.index_for_sample(12) == second.index_for_sample(12)


def test_rho_n_cutmix_returns_normalized_soft_label():
    dataset = ToyDataset()
    transform = RhoNCutMix(
        dataset,
        dataset.targets,
        num_classes=4,
        rho=0.8,
        donors=2,
        minimum_patch_size=2,
        seed=42,
    )
    sample = transform(0)
    assert sample.image.shape == dataset[0][0].shape
    assert torch.isclose(sample.soft_label.sum(), torch.tensor(1.0))
    assert sample.soft_label[0] == 0
    assert len(sample.donor_indices) == 2
