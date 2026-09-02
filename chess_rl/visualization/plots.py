"""
Scientific visualization utilities for training curves and sample efficiency curves.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt


def plot_training_curves(
    metrics_history: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Training Convergence Curves",
) -> None:
    """
    Plot total, policy, and value losses over iterations and save to disk.
    """
    if not metrics_history:
        return

    iterations = [m.get("iteration", idx) for idx, m in enumerate(metrics_history)]
    total_losses = [m.get("loss_total", 0.0) for m in metrics_history]
    val_losses = [m.get("loss_value", 0.0) for m in metrics_history]
    pol_losses = [m.get("loss_policy", 0.0) for m in metrics_history]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(iterations, total_losses, label="Total Loss", color="#1f77b4", linewidth=2)
    ax1.plot(iterations, val_losses, label="Value Loss (MSE)", color="#2ca02c", linestyle="--")
    ax1.plot(iterations, pol_losses, label="Policy Loss (CE)", color="#ff7f0e", linestyle=":")

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title(title)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
