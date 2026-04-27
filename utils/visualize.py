import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from typing import List, Dict

# Style

COLORS = {
    "fedavg":        "#e74c3c",
    "median":        "#3498db",
    "trimmed_mean":  "#2ecc71",
    "krum":          "#9b59b6",
    "reputation":    "#f39c12",
    "no_defense":    "#95a5a6",
}

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "#f8f9fa",
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.family":      "sans-serif",
    "font.size":        11,
}

plt.rcParams.update(STYLE)


def _label(config: dict) -> str:
    agg = config["aggregation"]
    atk = config["attack"]
    r   = int(config["malicious_ratio"] * 100)
    return f"{agg} ({atk}, {r}% malicious)"


# Plot 1: Accuracy Curves

def plot_accuracy_comparison(results_list: List[dict], save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Test Accuracy vs. Training Rounds", fontsize=14, fontweight="bold")

    for res in results_list:
        cfg   = res["config"]
        color = COLORS.get(cfg["aggregation"], "#333")
        label = _label(cfg)
        axes[0].plot(res["rounds"], res["accuracy"], color=color, linewidth=1.8, label=label)
        axes[1].plot(res["rounds"], res["asr"],      color=color, linewidth=1.8,
                     linestyle="--", label=label)

    axes[0].set_title("Global Model Accuracy")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].set_title("Attack Success Rate (ASR)")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("ASR")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# Plot 2: Reputation Score Evolution 

def plot_reputation_evolution(
    reputation_history: Dict[str, List[float]],
    malicious_ids: List[int],
    save_path: str,
    title: str = "Client Reputation Scores Over Rounds",
):
    fig, ax = plt.subplots(figsize=(10, 5))

    for cid_str, scores in reputation_history.items():
        cid     = int(cid_str)
        is_mal  = cid in malicious_ids
        color   = "#e74c3c" if is_mal else "#2ecc71"
        alpha   = 0.9 if is_mal else 0.6
        lw      = 2.0 if is_mal else 1.2
        linesty = "--" if is_mal else "-"
        ax.plot(range(len(scores)), scores,
                color=color, alpha=alpha, linewidth=lw, linestyle=linesty,
                label=f"Client {cid} ({'Malicious' if is_mal else 'Honest'})")

    ax.axhline(0.2, color="#c0392b", linewidth=1.5, linestyle=":", label="Exclusion Threshold")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Round")
    ax.set_ylabel("Reputation Score")
    ax.set_ylim(-0.05, 2.2)

    handles = [
        Line2D([0], [0], color="#e74c3c", lw=2, linestyle="--", label="Malicious Clients"),
        Line2D([0], [0], color="#2ecc71", lw=1.5, label="Honest Clients"),
        Line2D([0], [0], color="#c0392b", lw=1.5, linestyle=":", label="Exclusion Threshold (0.2)"),
    ]
    ax.legend(handles=handles, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# Plot 3: Final Accuracy Bar Chart

def plot_final_accuracy_bar(results_list: List[dict], save_path: str):
    methods = []
    accs    = []
    asrs    = []
    colors  = []

    for res in results_list:
        cfg = res["config"]
        methods.append(f"{cfg['aggregation']}\n({cfg['attack']}, {int(cfg['malicious_ratio']*100)}%)")
        accs.append(res["accuracy"][-1] if res["accuracy"] else 0)
        asrs.append(res["asr"][-1]      if res["asr"]      else 0)
        colors.append(COLORS.get(cfg["aggregation"], "#888"))

    x   = np.arange(len(methods))
    w   = 0.38
    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.6), 5))

    bars1 = ax.bar(x - w/2, accs, w, label="Test Accuracy (↑)", color=colors, alpha=0.85)
    bars2 = ax.bar(x + w/2, asrs, w, label="ASR (↓)",           color=colors, alpha=0.45,
                   edgecolor="black", linewidth=0.8)

    ax.set_title("Final Round: Test Accuracy vs. Attack Success Rate", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# Plot 4: Non-IID Label Distribution Heatmap

def plot_label_distribution(dist: np.ndarray, save_path: str, dataset: str = "MNIST"):
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(dist, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, ax=ax, label="Sample count")
    ax.set_title(f"Non-IID Label Distribution ({dataset}, Dirichlet α=0.5)", fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client ID")
    ax.set_xticks(range(dist.shape[1]))
    ax.set_yticks(range(dist.shape[0]))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# Plot 5: Convergence Comparison (reputation vs baselines) 

def plot_convergence_comparison(
    results_by_method: Dict[str, dict],
    attack: str,
    save_path: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Convergence Under {attack.replace('_', ' ').title()} Attack",
                 fontsize=13, fontweight="bold")

    for method, res in results_by_method.items():
        color = COLORS.get(method, "#888")
        axes[0].plot(res["rounds"], res["accuracy"], color=color, linewidth=2, label=method)
        axes[1].plot(res["rounds"], res["loss"],     color=color, linewidth=2, label=method)

    for ax, ylabel, title in zip(
        axes,
        ["Test Accuracy", "Test Loss"],
        ["Accuracy", "Loss"]
    ):
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# Load & Generate All Plots

def generate_all_plots(results_dir: str = "./results", plots_dir: str = "./plots"):
    os.makedirs(plots_dir, exist_ok=True)

    # Load all result files
    results = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(results_dir, fname)) as f:
                results.append(json.load(f))

    if not results:
        print("No results found.")
        return

    print(f"\nGenerating plots from {len(results)} result files...")

    # Plot 1: Accuracy and ASR curves
    plot_accuracy_comparison(results, os.path.join(plots_dir, "accuracy_comparison.png"))

    # Plot 2: Final accuracy bar chart
    plot_final_accuracy_bar(results, os.path.join(plots_dir, "final_accuracy_bar.png"))

    # Plot 3: Reputation evolution (find a reputation result)
    for res in results:
        if res["config"]["aggregation"] == "reputation" and res.get("reputation_history"):
            mal_ratio = res["config"]["malicious_ratio"]
            n_clients = res["config"]["num_clients"]
            mal_ids   = list(range(n_clients - int(n_clients * mal_ratio), n_clients))
            plot_reputation_evolution(
                res["reputation_history"], mal_ids,
                os.path.join(plots_dir, "reputation_evolution.png"),
                title=f"Reputation Evolution — {res['config']['attack']} attack"
            )
            break

    # Plot 4: Convergence by method for byzantine attack
    byz_methods = {}
    for res in results:
        cfg = res["config"]
        if cfg["attack"] == "byzantine" and cfg["malicious_ratio"] == 0.2:
            byz_methods[cfg["aggregation"]] = res
    if byz_methods:
        plot_convergence_comparison(
            byz_methods, "byzantine",
            os.path.join(plots_dir, "convergence_byzantine.png")
        )

    print(f"\nAll plots saved to {plots_dir}/")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results")
    p.add_argument("--plots_dir",   default="./plots")
    args = p.parse_args()
    generate_all_plots(args.results_dir, args.plots_dir)
