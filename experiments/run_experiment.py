import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from core.model import get_model
from core.data_partition import get_dataset, dirichlet_partition, get_client_loaders
from core.client import FLClient
from core.server import FLServer
from attacks.poisoning import (
    byzantine_attack, scale_attack,
    label_flip_client, backdoor_client, BackdoorDataset,
    GRADIENT_ATTACKS
)


# Experiment Config

SWEEP_CONFIGS = [
    # (aggregation,   attack,      malicious_ratio)
    ("fedavg",       "none",       0.0),
    ("fedavg",       "byzantine",  0.2),
    ("fedavg",       "byzantine",  0.3),
    ("median",       "byzantine",  0.2),
    ("trimmed_mean", "byzantine",  0.2),
    ("krum",         "byzantine",  0.2),
    ("reputation",   "byzantine",  0.2),
    ("reputation",   "byzantine",  0.3),
    ("fedavg",       "label_flip", 0.2),
    ("reputation",   "label_flip", 0.2),
    ("fedavg",       "backdoor",   0.2),
    ("reputation",   "backdoor",   0.2),
]


def run_experiment(
    dataset:          str   = "mnist",
    num_clients:      int   = 10,
    malicious_ratio:  float = 0.2,
    attack:           str   = "byzantine",
    aggregation:      str   = "reputation",
    rounds:           int   = 30,
    local_epochs:     int   = 2,
    lr:               float = 0.01,
    alpha:            float = 0.5,
    batch_size:       int   = 32,
    seed:             int   = 42,
    data_dir:         str   = "./data",
    results_dir:      str   = "./results",
) -> dict:
    """
    Run a single FL experiment and return metrics history.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} | Attack: {attack} | Agg: {aggregation}")
    print(f"Malicious: {malicious_ratio*100:.0f}% | Rounds: {rounds} | Device: {device}")
    print('='*60)

    # Data
    train_dataset, test_dataset = get_dataset(dataset, data_dir)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    client_indices = dirichlet_partition(train_dataset, num_clients, alpha=alpha, seed=seed)
    client_loaders = get_client_loaders(client_indices, train_dataset, batch_size)

    # Clients
    num_malicious  = max(0, int(num_clients * malicious_ratio))
    malicious_ids  = list(range(num_clients - num_malicious, num_clients))
    honest_ids     = [i for i in range(num_clients) if i not in malicious_ids]

    clients = []
    for cid in range(num_clients):
        c = FLClient(
            client_id=cid,
            dataloader=client_loaders[cid],
            device=device,
            local_epochs=local_epochs,
            lr=lr,
            is_malicious=cid in malicious_ids,
        )
        clients.append(c)

    # Apply data-poisoning attacks (label flip, backdoor)
    backdoor_test_loader = None
    if attack == "label_flip":
        for cid in malicious_ids:
            label_flip_client(clients[cid], source_label=1, target_label=7, batch_size=batch_size)

    elif attack == "backdoor":
        for cid in malicious_ids:
            backdoor_client(clients[cid], target_class=0, poison_rate=0.3, batch_size=batch_size)
        # Backdoor test set: all samples with trigger, labeled as target_class=0
        bd_test = BackdoorDataset(test_dataset, target_class=0, poison_rate=1.0, trigger_size=3)
        backdoor_test_loader = DataLoader(bd_test, batch_size=256, shuffle=False, num_workers=0)

    # Gradient attack functions
    attack_fn_map = {}
    if attack in GRADIENT_ATTACKS:
        fn = GRADIENT_ATTACKS[attack]()
        for cid in malicious_ids:
            attack_fn_map[cid] = fn

    # Server 
    global_model = get_model(dataset)
    use_rep = (aggregation == "reputation")
    server = FLServer(
        global_model=global_model,
        clients=clients,
        malicious_ids=malicious_ids,
        aggregation=aggregation,
        device=device,
        num_byzantine=num_malicious,
        trim_fraction=0.1,
        use_reputation=use_rep,
    )

    # Training Loop 
    results = {
        "config": {
            "dataset": dataset, "attack": attack, "aggregation": aggregation,
            "malicious_ratio": malicious_ratio, "num_clients": num_clients,
            "rounds": rounds, "alpha": alpha,
        },
        "rounds":       [],
        "accuracy":     [],
        "loss":         [],
        "asr":          [],
        "num_active":   [],
        "reputation_history": {},
    }

    for r in range(1, rounds + 1):
        t0 = time.time()
        metrics = server.run_round(
            round_id=r,
            attack_fn_map=attack_fn_map,
            client_fraction=1.0,
        )
        loss, acc, asr = server.evaluate(test_loader, backdoor_test_loader)
        metrics.test_accuracy = acc
        metrics.test_loss     = loss
        metrics.asr           = asr

        results["rounds"].append(r)
        results["accuracy"].append(round(acc, 4))
        results["loss"].append(round(loss, 4))
        results["asr"].append(round(asr, 4))
        results["num_active"].append(metrics.num_active)

        elapsed = time.time() - t0
        print(f"  Round {r:3d}/{rounds} | Acc: {acc:.4f} | Loss: {loss:.4f} "
              f"| ASR: {asr:.4f} | Active: {metrics.num_active} | {elapsed:.1f}s")

    # Blockchain audit
    if server.blockchain is not None:
        results["reputation_history"] = {
            str(k): v for k, v in server.blockchain.get_reputation_history().items()
        }
        results["chain_valid"] = server.blockchain.verify_chain()
        print(f"\n  Chain integrity: {'✓ VALID' if results['chain_valid'] else '✗ INVALID'}")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    fname = f"{results_dir}/{dataset}_{attack}_{aggregation}_{int(malicious_ratio*100)}pct.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {fname}")

    return results


def run_sweep(dataset="mnist", rounds=20, results_dir="./results"):
    """Run all configurations in SWEEP_CONFIGS."""
    all_results = []
    for agg, attack, ratio in SWEEP_CONFIGS:
        r = run_experiment(
            dataset=dataset,
            aggregation=agg,
            attack=attack,
            malicious_ratio=ratio,
            rounds=rounds,
            results_dir=results_dir,
        )
        all_results.append(r)
    return all_results


# CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BFT-FL Experiment Runner")
    parser.add_argument("--dataset",         default="mnist",       choices=["mnist", "cifar10"])
    parser.add_argument("--num_clients",     type=int,   default=10)
    parser.add_argument("--malicious_ratio", type=float, default=0.2)
    parser.add_argument("--attack",          default="byzantine",
                        choices=["none", "byzantine", "scale", "label_flip", "backdoor"])
    parser.add_argument("--aggregation",     default="reputation",
                        choices=["fedavg", "median", "trimmed_mean", "krum", "reputation"])
    parser.add_argument("--rounds",          type=int,   default=30)
    parser.add_argument("--local_epochs",    type=int,   default=2)
    parser.add_argument("--lr",              type=float, default=0.01)
    parser.add_argument("--alpha",           type=float, default=0.5,
                        help="Dirichlet non-IID parameter (lower = more heterogeneous)")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--data_dir",        default="./data")
    parser.add_argument("--results_dir",     default="./results")
    parser.add_argument("--sweep",           action="store_true",
                        help="Run full comparison sweep")

    args = parser.parse_args()

    if args.sweep:
        run_sweep(dataset=args.dataset, rounds=args.rounds, results_dir=args.results_dir)
    else:
        run_experiment(
            dataset=args.dataset,
            num_clients=args.num_clients,
            malicious_ratio=args.malicious_ratio,
            attack=args.attack,
            aggregation=args.aggregation,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            lr=args.lr,
            alpha=args.alpha,
            seed=args.seed,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
        )
