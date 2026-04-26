import hashlib
import json
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ──Block & Ledger

@dataclass
class ReputationBlock:
    """A single block in the reputation ledger."""
    round_id:         int
    timestamp:        float
    client_scores:    Dict[int, float]      # client_id → score this round
    consensus_norm:   float                  # L2 norm of the consensus update
    client_deviations: Dict[int, float]     # client_id → deviation from consensus
    prev_hash:        str
    block_hash:       str = field(default="", init=False)

    def __post_init__(self):
        self.block_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps({
            "round_id":          self.round_id,
            "timestamp":         self.timestamp,
            "client_scores":     {str(k): v for k, v in self.client_scores.items()},
            "consensus_norm":    self.consensus_norm,
            "prev_hash":         self.prev_hash,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def verify(self) -> bool:
        """Check integrity of this block."""
        return self.block_hash == self._compute_hash()


class ReputationBlockchain:
    """Simulated blockchain for persistent client reputation tracking. """

    GENESIS_SCORE = 1.0          # Starting reputation for all clients
    DECAY_RATE    = 0.1          # How quickly bad behavior reduces score
    REWARD_RATE   = 0.05         # How quickly good behavior increases score
    MAX_SCORE     = 2.0          # Cap on reputation
    MIN_SCORE     = 0.0          # Floor on reputation
    EXCLUSION_THRESHOLD = 0.2   # Below this → client excluded from aggregation

    def __init__(self, client_ids: List[int]):
        self.client_ids = client_ids
        self.chain: List[ReputationBlock] = []
        self.reputation: Dict[int, float] = {
            cid: self.GENESIS_SCORE for cid in client_ids
        }
        self.history: Dict[int, List[float]] = {cid: [self.GENESIS_SCORE] for cid in client_ids}
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = ReputationBlock(
            round_id=0,
            timestamp=time.time(),
            client_scores={cid: self.GENESIS_SCORE for cid in self.client_ids},
            consensus_norm=0.0,
            client_deviations={cid: 0.0 for cid in self.client_ids},
            prev_hash="0" * 64,
        )
        self.chain.append(genesis)

    def _flat(self, update: Dict[str, torch.Tensor]) -> np.ndarray:
        return torch.cat([v.flatten() for v in update.values()]).numpy()

    def compute_consensus(
        self,
        updates: Dict[int, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Compute coordinate-wise median as the trusted consensus update."""
        keys = list(next(iter(updates.values())).keys())
        all_updates = list(updates.values())
        consensus = {}
        for key in keys:
            stacked = torch.stack([u[key] for u in all_updates])
            consensus[key] = stacked.median(dim=0).values
        return consensus

    def update_reputations(
        self,
        round_id: int,
        updates: Dict[int, Dict[str, torch.Tensor]],
        participating_ids: List[int],
    ) -> Dict[int, float]:
        """
        Smart contract logic: update reputation scores based on deviation
        from the consensus update.
        
        """
        consensus = self.compute_consensus(updates)
        consensus_flat = self._flat(consensus)
        consensus_norm = float(np.linalg.norm(consensus_flat)) + 1e-9

        deviations: Dict[int, float] = {}

        for cid in participating_ids:
            client_flat = self._flat(updates[cid])
            # Normalized cosine-distance deviation
            cos_sim = float(
                np.dot(client_flat, consensus_flat) /
                (np.linalg.norm(client_flat) + 1e-9) /
                consensus_norm
            )
            l2_dev = float(np.linalg.norm(client_flat - consensus_flat)) / consensus_norm
            # Combined deviation score (higher = more suspicious)
            deviation = l2_dev * (1.0 - max(cos_sim, 0.0))
            deviations[cid] = deviation

        # Normalize deviations to [0,1] relative to the group
        dev_vals = np.array(list(deviations.values()))
        dev_min, dev_max = dev_vals.min(), dev_vals.max()
        dev_range = dev_max - dev_min + 1e-9

        round_scores: Dict[int, float] = {}

        for cid in participating_ids:
            norm_dev = (deviations[cid] - dev_min) / dev_range  # [0,1], 1=bad
            current  = self.reputation[cid]

            if norm_dev > 0.5:
                # Penalize: suspicious behavior
                new_score = current * (1.0 - self.DECAY_RATE * norm_dev)
            else:
                # Reward: consistent with consensus
                new_score = current + self.REWARD_RATE * (1.0 - norm_dev)

            new_score = float(np.clip(new_score, self.MIN_SCORE, self.MAX_SCORE))
            self.reputation[cid] = new_score
            round_scores[cid]    = new_score
            self.history[cid].append(new_score)

        # Non-participating clients get a small passive decay
        for cid in self.client_ids:
            if cid not in participating_ids:
                self.reputation[cid] = max(
                    self.MIN_SCORE,
                    self.reputation[cid] * 0.95
                )
                self.history[cid].append(self.reputation[cid])
                round_scores[cid] = self.reputation[cid]
                deviations[cid]   = 0.0

        # Append block to chain
        block = ReputationBlock(
            round_id=round_id,
            timestamp=time.time(),
            client_scores=dict(round_scores),
            consensus_norm=float(consensus_norm),
            client_deviations=dict(deviations),
            prev_hash=self.chain[-1].block_hash,
        )
        self.chain.append(block)
        return dict(self.reputation)

    def get_active_clients(self, client_ids: List[int]) -> List[int]:
        """Return clients with reputation above the exclusion threshold."""
        return [
            cid for cid in client_ids
            if self.reputation.get(cid, self.GENESIS_SCORE) >= self.EXCLUSION_THRESHOLD
        ]

    def get_reputation_weights(self, client_ids: List[int]) -> List[float]:
        """Return normalized reputation weights for the given client IDs."""
        scores = [self.reputation.get(cid, self.GENESIS_SCORE) for cid in client_ids]
        return scores

    def verify_chain(self) -> bool:
        """Verify integrity of the entire chain."""
        for i in range(1, len(self.chain)):
            if not self.chain[i].verify():
                return False
            if self.chain[i].prev_hash != self.chain[i - 1].block_hash:
                return False
        return True

    def get_audit_log(self) -> List[Dict]:
        """Return human-readable audit log of all blocks."""
        return [
            {
                "round":      b.round_id,
                "timestamp":  b.timestamp,
                "scores":     b.client_scores,
                "deviations": b.client_deviations,
                "hash":       b.block_hash[:12] + "...",
            }
            for b in self.chain
        ]

    def get_reputation_history(self) -> Dict[int, List[float]]:
        """Return full per-client reputation history."""
        return dict(self.history)
