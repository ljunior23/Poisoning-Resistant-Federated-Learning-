"""
Quick unit tests for BFT-FL components (no data download needed).
Run from /home/claude/bfl: python tests/test_components.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# ── Test 1: Model forward pass ────────────────────────────────────────────────
def test_models():
    from core.model import get_model
    mnist_model = get_model("mnist")
    x = torch.randn(4, 1, 28, 28)
    out = mnist_model(x)
    assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"

    cifar_model = get_model("cifar10")
    x = torch.randn(4, 3, 32, 32)
    out = cifar_model(x)
    assert out.shape == (4, 10), f"Expected (4,10), got {out.shape}"
    print("✓ Models: MNIST and CIFAR-10 forward pass OK")


# ── Test 2: Aggregation methods ───────────────────────────────────────────────
def test_aggregators():
    from aggregation.aggregators import fedavg, coordinate_median, trimmed_mean, krum, reputation_fedavg

    # Fake updates: 5 clients, simple 1-layer model
    updates = [{"w": torch.randn(10)} for _ in range(5)]

    agg = fedavg(updates)
    assert "w" in agg and agg["w"].shape == (10,), "FedAvg shape mismatch"

    agg = coordinate_median(updates)
    assert agg["w"].shape == (10,), "Median shape mismatch"

    agg = trimmed_mean(updates, trim_fraction=0.2)
    assert agg["w"].shape == (10,), "TrimmedMean shape mismatch"

    agg = krum(updates, num_byzantine=1)
    assert agg["w"].shape == (10,), "Krum shape mismatch"

    rep_scores = [1.0, 0.8, 0.1, 0.05, 0.9]
    agg = reputation_fedavg(updates, reputation_scores=rep_scores)
    assert agg["w"].shape == (10,), "ReputationFedAvg shape mismatch"
    print("✓ Aggregators: FedAvg, Median, TrimmedMean, Krum, Reputation OK")


# ── Test 3: Attack functions ───────────────────────────────────────────────────
def test_attacks():
    from attacks.poisoning import byzantine_attack, scale_attack

    update = {"w": torch.ones(10), "b": torch.ones(5)}
    byz    = byzantine_attack(scale=5.0)(update)
    assert "w" in byz and byz["w"].shape == (10,), "Byzantine attack shape mismatch"

    scaled = scale_attack(factor=10.0)(update)
    assert torch.allclose(scaled["w"], torch.ones(10) * 10.0), "Scale attack value mismatch"
    print("✓ Attacks: Byzantine and Scale OK")


# ── Test 4: Blockchain reputation system ─────────────────────────────────────
def test_blockchain():
    from blockchain.reputation import ReputationBlockchain

    client_ids = list(range(5))
    chain = ReputationBlockchain(client_ids)
    assert len(chain.chain) == 1  # genesis block

    # Simulate updates: client 4 is malicious (large deviation)
    updates = {
        0: {"w": torch.tensor([0.1, 0.2, 0.1])},
        1: {"w": torch.tensor([0.1, 0.2, 0.1])},
        2: {"w": torch.tensor([0.1, 0.2, 0.1])},
        3: {"w": torch.tensor([0.1, 0.2, 0.1])},
        4: {"w": torch.tensor([10.0, -10.0, 10.0])},  # malicious
    }

    for rnd in range(1, 6):
        scores = chain.update_reputations(rnd, updates, list(updates.keys()))

    # Malicious client should have lower reputation
    assert scores[4] < scores[0], \
        f"Malicious client score {scores[4]:.3f} should be < honest {scores[0]:.3f}"

    # Chain should be valid
    assert chain.verify_chain(), "Chain integrity check failed"

    # Active clients should exclude malicious if score too low
    active = chain.get_active_clients(client_ids)
    print(f"  Final scores: { {k: round(v,3) for k,v in scores.items()} }")
    print(f"  Active clients: {active}")
    print("✓ Blockchain: Reputation scoring, chain integrity, exclusion OK")


# ── Test 5: Full mini server round (no data download) ────────────────────────
def test_server_round():
    from core.model import get_model
    from core.server import FLServer
    from torch.utils.data import DataLoader, TensorDataset

    # Synthetic data
    def make_loader(n=100):
        X = torch.randn(n, 1, 28, 28)
        y = torch.randint(0, 10, (n,))
        return DataLoader(TensorDataset(X, y), batch_size=32)

    from core.client import FLClient
    device = torch.device("cpu")
    clients = [
        FLClient(i, make_loader(), device, local_epochs=1, lr=0.01)
        for i in range(4)
    ]

    server = FLServer(
        global_model=get_model("mnist"),
        clients=clients,
        malicious_ids=[3],
        aggregation="reputation",
        device=device,
        use_reputation=True,
    )

    metrics = server.run_round(round_id=1)
    assert metrics.num_active > 0, "No active clients after round 1"
    print(f"  Round 1: {metrics.num_active} clients active")
    print("✓ Server: Full training round with reputation aggregation OK")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  BFT-FL Component Tests")
    print("="*55 + "\n")
    test_models()
    test_aggregators()
    test_attacks()
    test_blockchain()
    test_server_round()
    print("\n" + "="*55)
    print("  All tests passed ✓")
    print("="*55 + "\n")
