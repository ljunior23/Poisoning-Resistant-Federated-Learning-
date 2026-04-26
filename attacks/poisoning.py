import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Callable


# Byzantine Attack 

def byzantine_attack(scale: float = 10.0) -> Callable:
    """
    Replace gradient update with large random Gaussian noise.

    Args:
        scale: Noise magnitude multiplier.
    """
    def attack(update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            name: torch.randn_like(tensor) * scale
            for name, tensor in update.items()
        }
    return attack


# Scale Attack

def scale_attack(factor: float = 20.0) -> Callable:
    """
    Amplify honest update by a large factor to overwhelm aggregation.

    Args:
        factor: Amplification factor.
    """
    def attack(update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {name: tensor * factor for name, tensor in update.items()}
    return attack


# Label-Flip Attack

class LabelFlipDataset(Dataset):
    """Wraps a dataset and flips source_label → target_label."""

    def __init__(self, dataset, source_label: int = 1, target_label: int = 7):
        self.dataset      = dataset
        self.source_label = source_label
        self.target_label = target_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y_int = int(y)
        if y_int == self.source_label:
            y_int = self.target_label
        return x, torch.tensor(y_int, dtype=torch.long)


def label_flip_client(
    client,
    source_label: int = 1,
    target_label: int = 7,
    batch_size: int = 32,
):
    """
    Return a modified client whose dataloader uses flipped labels.
    Modifies client.dataloader in-place.
    """
    original_dataset = client.dataloader.dataset
    flipped_dataset  = LabelFlipDataset(original_dataset, source_label, target_label)
    client.dataloader = DataLoader(
        flipped_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return client


# Backdoor Attack 

class BackdoorDataset(Dataset):
    """
    Injects a small trigger pattern (white square) into a fraction of samples
    and relabels them as target_class.

    """

    def __init__(
        self,
        dataset,
        target_class: int = 0,
        poison_rate: float = 0.3,
        trigger_size: int = 3,
        trigger_pos: tuple = (0, 0),
    ):
        self.dataset      = dataset
        self.target_class = target_class
        self.poison_rate  = poison_rate
        self.trigger_size = trigger_size
        self.trigger_pos  = trigger_pos

        n = len(dataset)
        self.poisoned_indices = set(
            np.random.choice(n, int(n * poison_rate), replace=False).tolist()
        )

    def _inject_trigger(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        r, c = self.trigger_pos
        t    = self.trigger_size
        # x shape: (C, H, W)
        x[:, r: r + t, c: c + t] = 1.0
        return x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if idx in self.poisoned_indices:
            x = self._inject_trigger(x)
            y = self.target_class
        return x, torch.tensor(int(y), dtype=torch.long)


def backdoor_client(
    client,
    target_class: int = 0,
    poison_rate: float = 0.3,
    trigger_size: int = 3,
    batch_size: int = 32,
):
    """Return a modified client whose dataloader contains backdoor poisoning."""
    original_dataset = client.dataloader.dataset
    poisoned_dataset = BackdoorDataset(
        original_dataset,
        target_class=target_class,
        poison_rate=poison_rate,
        trigger_size=trigger_size,
    )
    client.dataloader = DataLoader(
        poisoned_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return client


# Attack Registry 

GRADIENT_ATTACKS = {
    "byzantine":  byzantine_attack,
    "scale":      scale_attack,
}

DATA_ATTACKS = {
    "label_flip": label_flip_client,
    "backdoor":   backdoor_client,
}
