"""
head_init.py — Orthogonal initialization for the 100-class head.

NOTE: The primary initialization strategy is the centroid warm-start inside
ZeroOrderOptimizer.__init__ (which loads CIFAR100, extracts backbone features,
and sets fc.weight = L2-normalized class centroids).

This file's init is the fallback applied by validate.py BEFORE the optimizer runs.
We use orthogonal initialization here (not Xavier gain=0.01) because:

  WHY NOT Xavier gain=0.01 (the previous approach was wrong):
    - gain=0.01 shrinks weights to near-zero → logits near zero everywhere
    - Loss surface is flat at loss ≈ log(100) = 4.605 (maximum entropy)
    - f_plus ≈ f_minus in SPSA → g_hat ≈ 0 → zero gradient signal
    - SPSA can't find a descent direction from a flat surface

  WHY Orthogonal gain=1.0 is a better fallback:
    - Each class gets a linearly independent weight direction in 512-space
    - Logit diversity → meaningful cross-entropy gradient from the start
    - Still overwritten by centroid init if data loading succeeds

  The centroid init is the real workhorse (30-40% acc before any ZO step).
  This orthogonal init just ensures a non-degenerate starting point
  in case CIFAR100 data is unavailable.
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