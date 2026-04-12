import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict


def get_dataset(dataset_name: str, data_dir: str = "./data"):
    """Download and return train/test datasets."""
    if dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
        test  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif dataset_name.lower() == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train = datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform_train)
        test  = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train, test


def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float = 0.5,
    min_samples: int = 10,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Partition dataset indices across clients using a Dirichlet distribution."""
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    class_indices = {c: np.where(targets == c)[0].tolist() for c in range(num_classes)}

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Convert to actual counts
        proportions = (proportions * len(indices)).astype(int)
        # Fix rounding: assign leftover to first client
        proportions[0] += len(indices) - proportions.sum()

        idx = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(indices[idx: idx + count])
            idx += count

    # Ensure minimum samples per client by redistributing
    for client_id in range(num_clients):
        if len(client_indices[client_id]) < min_samples:
            # Borrow from client with most samples
            donor = max(client_indices, key=lambda k: len(client_indices[k]))
            borrow = client_indices[donor][:min_samples]
            client_indices[client_id].extend(borrow)
            client_indices[donor] = client_indices[donor][min_samples:]

    return client_indices


def get_client_loaders(
    client_indices: Dict[int, List[int]],
    dataset,
    batch_size: int = 32,
) -> Dict[int, DataLoader]:
    """Create DataLoaders for each client."""
    loaders = {}
    for client_id, indices in client_indices.items():
        subset = Subset(dataset, indices)
        loaders[client_id] = DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=0
        )
    return loaders


def get_label_distribution(client_indices: Dict[int, List[int]], dataset) -> np.ndarray:
    """Return (num_clients x num_classes) matrix of label counts."""
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    dist = np.zeros((len(client_indices), num_classes), dtype=int)
    for cid, indices in client_indices.items():
        for idx in indices:
            dist[cid, targets[idx]] += 1
    return dist
