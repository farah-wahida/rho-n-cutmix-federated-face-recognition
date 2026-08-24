from collections import OrderedDict

import torch

from privacy_pipeline.federated import fedavg, soft_cross_entropy


def test_fedavg_uses_sample_count_weights():
    states = [
        OrderedDict(weight=torch.tensor([1.0, 3.0])),
        OrderedDict(weight=torch.tensor([5.0, 7.0])),
    ]
    averaged = fedavg(states, sample_counts=[1, 3])
    assert torch.allclose(averaged["weight"], torch.tensor([4.0, 6.0]))


def test_soft_cross_entropy_accepts_mixture_targets():
    logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.25, 0.75]])
    loss = soft_cross_entropy(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
