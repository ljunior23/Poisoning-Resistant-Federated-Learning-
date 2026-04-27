# Poisoning-Resistant Federated Learning with Blockchain-Based Reputation Scoring

**CIS-5700 — University of Michigan-Dearborn**
**Authors:** Kumi Acheampong & Venkata Manaswitha Sunkara

---

## Quick Start

```bash
pip install -r requirements.txt

# Single experiment
python experiments/run_experiment.py \
    --dataset mnist \
    --attack byzantine \
    --aggregation reputation \
    --malicious_ratio 0.2 \
    --rounds 30

# Full comparison sweep (all methods × attacks × ratios)
python experiments/run_experiment.py --sweep --dataset mnist --rounds 20

# Generate plots from saved results
python utils/visualize.py --results_dir ./results --plots_dir ./plots

# Run unit tests
python tests/test_components.py
```

---

## Project Structure

```
bfl/
├── core/
│   ├── model.py            # MNISTNet (CNN), CIFAR10Net (deep CNN)
│   ├── data_partition.py   # Dirichlet non-IID partition + DataLoaders
│   ├── client.py           # FLClient: local training, attack hooks
│   └── server.py           # FLServer: round orchestration, evaluation
├── attacks/
│   └── poisoning.py        # Byzantine, Scale, LabelFlip, Backdoor
├── aggregation/
│   └── aggregators.py      # FedAvg, Median, TrimmedMean, Krum, ReputationFedAvg
├── blockchain/
│   └── reputation.py       # ReputationBlockchain + ReputationBlock (SHA-256)
├── experiments/
│   └── run_experiment.py   # CLI + SWEEP_CONFIGS
├── utils/
│   └── visualize.py        # Matplotlib plotting (5 plot types)
└── tests/
    └── test_components.py  # Unit tests
```

---

## Architecture

### Non-IID Data Partitioning (`core/data_partition.py`)
Uses a Dirichlet distribution with concentration parameter α to partition data across N clients. Lower α produces more heterogeneous (realistic) splits.

### FL Clients (`core/client.py`)
Each `FLClient` performs local SGD for `local_epochs` rounds starting from the global model, then returns the parameter delta (local − global). Malicious clients optionally apply an `attack_fn` to their update before submission.

### Poisoning Attacks (`attacks/poisoning.py`)
| Attack | Type | Description |
|---|---|---|
| Byzantine | Gradient | Replace update with scaled Gaussian noise |
| Scale | Gradient | Amplify update by large factor (×20) |
| Label Flip | Data | Flip source label → target label during training |
| Backdoor | Data | Inject pixel trigger, remap triggered samples to target class |

### Aggregation Methods (`aggregation/aggregators.py`)
| Method | Defense Strategy |
|---|---|
| FedAvg | Baseline — no defense |
| Coordinate Median | Byzantine-robust up to <50% |
| Trimmed Mean | Clips top/bottom fraction |
| Krum / Multi-Krum | Selects least-deviating update |
| **Reputation FedAvg** | **Weighted by blockchain reputation score** |

### Blockchain Reputation (`blockchain/reputation.py`)
- Each round produces a `ReputationBlock` storing client scores, deviations, and SHA-256 hashes.
- Reputation is updated via a **smart contract** rule: clients whose L2 deviation from the median exceeds the group median are penalized; conforming clients are rewarded.
- Clients below `EXCLUSION_THRESHOLD = 0.2` are excluded from aggregation entirely.
- `verify_chain()` checks hash-chain integrity after all rounds.

---

## Experimental Sweep

The `SWEEP_CONFIGS` list in `run_experiment.py` covers 12 conditions:

| Aggregation | Attack | Malicious % |
|---|---|---|
| FedAvg | None | 0% |
| FedAvg | Byzantine | 20%, 30% |
| Median | Byzantine | 20% |
| Trimmed Mean | Byzantine | 20% |
| Krum | Byzantine | 20% |
| **Reputation** | **Byzantine** | **20%, 30%** |
| FedAvg | Label Flip | 20% |
| Reputation | Label Flip | 20% |
| FedAvg | Backdoor | 20% |
| Reputation | Backdoor | 20% |

Datasets: MNIST and CIFAR-10. Dirichlet α = 0.5. 10 clients per experiment.

---

## Contribution Split

| Component | Owner |
|---|---|
| FL framework (FedAvg + robust aggregation) | Kumi |
| Non-IID data partitioning | Kumi |
| Poisoning attack simulation | Kumi |
| Reputation-weighted aggregation integration | Kumi |
| Blockchain logging & dynamic reputation scoring | Manaswitha |
| Experimental evaluation & comparative analysis | Manaswitha |
| Visualization & report preparation | Manaswitha |
