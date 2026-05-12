"""
head_init.py — Final layer initialization.
"""


import torch.nn as nn


def init_last_layer(layer: nn.Linear) -> None:
    """Initialize the 100-class head in-place with orthogonal weights.

    Uses gain=1.0 (unit-scale orthogonal matrix, not shrunk).
    Bias initialized to zero.

    This is overwritten by ZeroOrderOptimizer._centroid_init() on the first
    optimizer construction, which provides a much better starting point.
    """
    nn.init.orthogonal_(layer.weight, gain=1.0)
    nn.init.zeros_(layer.bias)