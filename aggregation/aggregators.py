"""
Aggregation strategies for Federated Learning.

Implemented:
  - FedAvg           : Standard weighted average.
  - Median           : Coordinate-wise median (Byzantine-robust).
  - TrimmedMean      : Trimmed mean (clips top/bottom fraction).
  - Krum             : Selects the update closest to its neighbors.
  - ReputationFedAvg : Reputation-score-weighted aggregation (our method).
"""
import torch
import numpy as np
from typing import List, Dict, Tuple


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _stack_updates(updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of update dicts → {param: (N, *shape)} tensor dict."""
    keys = list(updates[0].keys())
    return {k: torch.stack([u[k] for u in updates]) for k in keys}


def _flat(update: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten an update dict to a 1-D vector."""
    return torch.cat([v.flatten() for v in update.values()])


# ─── FedAvg ──────────────────────────────────────────────────────────────────

def fedavg(
    updates: List[Dict[str, torch.Tensor]],
    weights: List[float] = None,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Weighted average of client updates.

    Args:
        updates: List of parameter-update dicts from clients.
        weights: Per-client data-size weights (uniform if None).
    """
    n = len(updates)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    aggregated = {}
    for key in updates[0].keys():
        aggregated[key] = sum(w * u[key] for w, u in zip(weights, updates))
    return aggregated


# ─── Coordinate-wise Median ──────────────────────────────────────────────────

def coordinate_median(
    updates: List[Dict[str, torch.Tensor]],
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Coordinate-wise median — robust to up to <50% Byzantine clients."""
    stacked = _stack_updates(updates)
    return {k: stacked[k].median(dim=0).values for k in stacked}


# ─── Trimmed Mean ─────────────────────────────────────────────────────────────

def trimmed_mean(
    updates: List[Dict[str, torch.Tensor]],
    trim_fraction: float = 0.1,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise trimmed mean: removes top and bottom `trim_fraction`
    of values before averaging.

    Args:
        trim_fraction: Fraction of clients to trim from each end (0–0.5).
    """
    n = len(updates)
    k = max(1, int(trim_fraction * n))
    stacked = _stack_updates(updates)

    aggregated = {}
    for key, vals in stacked.items():
        # vals: (N, *shape)
        original_shape = vals.shape[1:]
        flat = vals.reshape(n, -1)           # (N, D)
        sorted_vals, _ = torch.sort(flat, dim=0)
        trimmed = sorted_vals[k: n - k]     # (N-2k, D)
        aggregated[key] = trimmed.mean(dim=0).reshape(original_shape)
    return aggregated


# ─── Krum ─────────────────────────────────────────────────────────────────────

def krum(
    updates: List[Dict[str, torch.Tensor]],
    num_byzantine: int = 0,
    multi_krum: int = 1,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Krum aggregation: selects the update(s) with smallest sum of squared
    distances to its nearest neighbors.

    Args:
        num_byzantine: Assumed number of Byzantine clients (f).
        multi_krum:    Number of updates to select and average (Multi-Krum).
                       Set to 1 for standard Krum.
    """
    n = len(updates)
    f = num_byzantine
    m = n - f - 2  # Number of nearest neighbors to consider

    if m <= 0:
        # Fall back to FedAvg if not enough clients
        return fedavg(updates)

    flat_updates = torch.stack([_flat(u) for u in updates])  # (N, D)

    # Pairwise squared distances
    dists = torch.cdist(flat_updates, flat_updates, p=2) ** 2  # (N, N)

    scores = []
    for i in range(n):
        row = dists[i].clone()
        row[i] = float("inf")
        nearest = torch.topk(row, m, largest=False).values
        scores.append(nearest.sum().item())

    selected_ids = sorted(range(n), key=lambda i: scores[i])[:multi_krum]
    selected_updates = [updates[i] for i in selected_ids]
    return fedavg(selected_updates)


# ─── Reputation-Weighted FedAvg (Our Method) ─────────────────────────────────

def reputation_fedavg(
    updates: List[Dict[str, torch.Tensor]],
    reputation_scores: List[float],
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Reputation-score-weighted aggregation.
    Clients with lower reputation scores contribute less to the global model.

    Args:
        updates:           List of parameter-update dicts.
        reputation_scores: Per-client reputation scores (non-negative).
    """
    scores = np.array(reputation_scores, dtype=float)
    scores = np.maximum(scores, 0.0)   # Clamp negatives to zero
    total  = scores.sum()

    if total < 1e-9:
        # All reputations zeroed — fall back to uniform average
        return fedavg(updates)

    weights = (scores / total).tolist()
    return fedavg(updates, weights=weights)


# ─── Aggregation Registry ─────────────────────────────────────────────────────

AGGREGATION_METHODS = {
    "fedavg":          fedavg,
    "median":          coordinate_median,
    "trimmed_mean":    trimmed_mean,
    "krum":            krum,
    "reputation":      reputation_fedavg,
}
