import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

from core.client import FLClient
from aggregation.aggregators import AGGREGATION_METHODS, fedavg
from blockchain.reputation import ReputationBlockchain


@dataclass
class RoundMetrics:
    round_id:         int
    test_accuracy:    float
    test_loss:        float
    asr:              float    # Attack Success Rate (backdoor/label-flip)
    num_active:       int      # Clients used in aggregation
    num_excluded:     int      # Clients excluded by reputation
    reputation_scores: Dict[int, float] = field(default_factory=dict)


class FLServer:
    """Central FL server."""

    def __init__(
        self,
        global_model:   nn.Module,
        clients:        List[FLClient],
        malicious_ids:  List[int],
        aggregation:    str = "fedavg",
        device:         torch.device = torch.device("cpu"),
        num_byzantine:  int = 0,
        trim_fraction:  float = 0.1,
        use_reputation: bool = False,
    ):
        self.global_model   = copy.deepcopy(global_model).to(device)
        self.clients        = clients
        self.malicious_ids  = set(malicious_ids)
        self.aggregation    = aggregation
        self.device         = device
        self.num_byzantine  = num_byzantine
        self.trim_fraction  = trim_fraction
        self.use_reputation = use_reputation

        self.client_ids = [c.client_id for c in clients]
        self.history: List[RoundMetrics] = []

        if use_reputation or aggregation == "reputation":
            self.blockchain = ReputationBlockchain(self.client_ids)
        else:
            self.blockchain = None

    def _apply_update(
        self,
        update: Dict[str, torch.Tensor],
        lr: float = 1.0,
    ):
        """Apply aggregated update to the global model."""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in update:
                    param.data.add_(update[name].to(self.device) * lr)

    def _get_client_data_sizes(self) -> Dict[int, int]:
        return {c.client_id: len(c.dataloader.dataset) for c in self.clients}

    def run_round(
        self,
        round_id:      int,
        attack_fn_map: Dict[int, Callable] = None,
        client_fraction: float = 1.0,
    ) -> RoundMetrics:
        """Execute one FL training round."""
        if attack_fn_map is None:
            attack_fn_map = {}

        # Sample clients
        n_select = max(1, int(len(self.clients) * client_fraction))
        selected = np.random.choice(len(self.clients), n_select, replace=False)
        selected_clients = [self.clients[i] for i in selected]

        # Filter by reputation gating
        if self.blockchain is not None:
            selected_ids = [c.client_id for c in selected_clients]
            active_ids   = set(self.blockchain.get_active_clients(selected_ids))
            selected_clients = [c for c in selected_clients if c.client_id in active_ids]
        else:
            active_ids = {c.client_id for c in selected_clients}

        # Collect updates
        updates: Dict[int, Dict[str, torch.Tensor]] = {}
        for client in selected_clients:
            attack_fn = attack_fn_map.get(client.client_id)
            update    = client.train(self.global_model, attack_fn=attack_fn)
            updates[client.client_id] = update

        if not updates:
            return RoundMetrics(round_id, 0, 0, 0, 0, len(selected_clients))

        update_list = list(updates.values())
        participating_ids = list(updates.keys())

        # Update blockchain reputation
        rep_scores = {}
        if self.blockchain is not None:
            rep_scores = self.blockchain.update_reputations(
                round_id, updates, participating_ids
            )

        # Aggregation
        agg_kwargs = {
            "num_byzantine": self.num_byzantine,
            "trim_fraction":  self.trim_fraction,
        }

        if self.aggregation == "reputation":
            weights = self.blockchain.get_reputation_weights(participating_ids)
            agg_update = AGGREGATION_METHODS["reputation"](
                update_list, reputation_scores=weights
            )
        else:
            agg_update = AGGREGATION_METHODS[self.aggregation](update_list, **agg_kwargs)

        # Apply to global model
        self._apply_update(agg_update)

        # Record metrics placeholder (populated by caller after evaluation)
        num_excluded = len(active_ids) - len(participating_ids) if self.blockchain else 0

        metrics = RoundMetrics(
            round_id=round_id,
            test_accuracy=0.0,
            test_loss=0.0,
            asr=0.0,
            num_active=len(participating_ids),
            num_excluded=num_excluded,
            reputation_scores=rep_scores,
        )
        self.history.append(metrics)
        return metrics

    def evaluate(
        self,
        test_loader,
        backdoor_loader=None,
    ) -> tuple:
        """Evaluate global model on test data."""
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(self.device), y.to(self.device)
                out  = self.global_model(X)
                total_loss += criterion(out, y).item() * y.size(0)
                correct    += (out.argmax(1) == y).sum().item()
                total      += y.size(0)

        test_loss = total_loss / total
        test_acc  = correct / total

        # Attack Success Rate
        asr = 0.0
        if backdoor_loader is not None:
            bd_correct, bd_total = 0, 0
            with torch.no_grad():
                for X, y in backdoor_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    out  = self.global_model(X)
                    bd_correct += (out.argmax(1) == y).sum().item()
                    bd_total   += y.size(0)
            asr = bd_correct / bd_total if bd_total > 0 else 0.0

        return test_loss, test_acc, asr
