"""Distance primitives used by post_analysis.py.

Provides cosine_distance and linear_cka_matrix, extracted from the
source analysis module. These are the only symbols from the original
boundary-disruption module that post_analysis.py requires.

This module requires torch. If torch is unavailable, import will fail
at the same point that post_analysis.py would have failed.
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def cosine_distance(h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    """Centered cosine distance: 1 - cosine_similarity(h1 - mean, h2 - mean)."""
    h1 = h1.float()
    h2 = h2.float()
    h1c = h1 - h1.mean()
    h2c = h2 - h2.mean()
    sim = F.cosine_similarity(h1c.unsqueeze(0), h2c.unsqueeze(0))
    return 1.0 - sim.squeeze()


def linear_cka_matrix(X: torch.Tensor, Y: torch.Tensor) -> float:
    """True linear CKA over sample matrices.

    X, Y : [n_samples, hidden_dim]

    Computes HSIC(XX^T, YY^T) / sqrt(HSIC(XX^T,XX^T) * HSIC(YY^T,YY^T))
    using the unbiased, double-centered Gram matrix estimator.
    """
    logger.debug("linear_cka_matrix: X=%s, Y=%s", list(X.shape), list(Y.shape))
    n = X.shape[0]
    if n < 2:
        return 0.0

    X = X.float()
    Y = Y.float()

    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    K = X @ X.T
    L = Y @ Y.T

    H = (
        torch.eye(n, dtype=X.dtype, device=X.device)
        - torch.ones(n, n, dtype=X.dtype, device=X.device) / n
    )
    Kc = H @ K @ H
    Lc = H @ L @ H

    hsic_KL = (Kc * Lc).sum() / (n - 1) ** 2
    hsic_KK = (Kc * Kc).sum() / (n - 1) ** 2
    hsic_LL = (Lc * Lc).sum() / (n - 1) ** 2

    denom = torch.sqrt(hsic_KK * hsic_LL)
    if denom < 1e-10:
        return 0.0
    return float(hsic_KL / denom)
