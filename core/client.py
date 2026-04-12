import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import numpy as np


class FLClient:
    """A single FL client responsible for local training."""

    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        device: torch.device,
        local_epochs: int = 2,
        lr: float = 0.01,
        is_malicious: bool = False,
    ):
        self.client_id    = client_id
        self.dataloader   = dataloader
        self.device       = device
        self.local_epochs = local_epochs
        self.lr           = lr
        self.is_malicious = is_malicious

    def train(
        self,
        global_model: nn.Module,
        attack_fn=None,
    ) -> Dict[str, torch.Tensor]:
        """Perform local training starting from global_model weights."""
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.local_epochs):
            for X, y in self.dataloader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                optimizer.step()

        # Compute update = local_weights - global_weights
        global_params = dict(global_model.named_parameters())
        update = {}
        for name, param in model.named_parameters():
            update[name] = (param.data.clone() - global_params[name].data.clone()).cpu()

        # Apply attack if malicious
        if self.is_malicious and attack_fn is not None:
            update = attack_fn(update)

        return update

    def evaluate(self, model: nn.Module, criterion=None) -> Tuple[float, float]:
        """Return (loss, accuracy) on local data."""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        model.eval()
        model.to(self.device)
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X, y in self.dataloader:
                X, y = X.to(self.device), y.to(self.device)
                out  = model(X)
                total_loss += criterion(out, y).item() * y.size(0)
                correct    += (out.argmax(1) == y).sum().item()
                total      += y.size(0)
        return total_loss / total, correct / total
