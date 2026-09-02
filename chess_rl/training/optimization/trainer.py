"""
Optimization and network training loop manager.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.optim as optim

from chess_rl.agents.networks.base import BasePolicyValueNet
from chess_rl.training.optimization.loss import AlphaZeroLoss
from chess_rl.training.replay_buffer.base import BaseReplayBuffer
from chess_rl.config.schema import OptimizerConfig, TrainingConfig


class Trainer:
    """
    Manages gradient updates and optimization steps for the Policy-Value neural network.
    """

    def __init__(
        self,
        network: BasePolicyValueNet,
        optimizer_config: OptimizerConfig,
        training_config: TrainingConfig,
        device: torch.device = torch.device("cpu"),
    ):
        self.network = network.to(device)
        self.device = device
        self.opt_config = optimizer_config
        self.train_config = training_config

        # Initialize PyTorch optimizer
        if optimizer_config.optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.network.parameters(),
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.weight_decay,
                betas=tuple(optimizer_config.betas),
                eps=optimizer_config.eps,
            )
        elif optimizer_config.optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(
                self.network.parameters(),
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.weight_decay,
                eps=optimizer_config.eps,
            )
        else:
            self.optimizer = optim.SGD(
                self.network.parameters(),
                lr=optimizer_config.learning_rate,
                weight_decay=optimizer_config.weight_decay,
                momentum=0.9,
            )

        self.criterion = AlphaZeroLoss(optimizer_config.loss_weights)
        self.total_steps = 0

    def train_epoch(self, replay_buffer: BaseReplayBuffer, num_batches: int = 10) -> Dict[str, float]:
        """
        Train the network on a set of mini-batches sampled from the replay buffer.

        Args:
            replay_buffer: Experience buffer with stored transitions.
            num_batches: Number of gradient steps to execute.

        Returns:
            Dictionary of average loss metrics across batches.
        """
        batch_size = self.train_config.replay_buffer.batch_size
        if len(replay_buffer) < batch_size:
            return {"loss_total": 0.0, "loss_value": 0.0, "loss_policy": 0.0}

        self.network.train()
        total_loss_accum = 0.0
        val_loss_accum = 0.0
        pol_loss_accum = 0.0
        actual_batches = 0

        for _ in range(num_batches):
            try:
                obs_np, pis_np, vals_np, masks_np = replay_buffer.sample_batch(batch_size)
            except ValueError:
                break

            obs_t = torch.from_numpy(obs_np).to(self.device)
            pis_t = torch.from_numpy(pis_np).to(self.device)
            vals_t = torch.from_numpy(vals_np).to(self.device)
            masks_t = torch.from_numpy(masks_np).to(self.device) if masks_np is not None else None

            self.optimizer.zero_grad()
            logits, val_pred = self.network(obs_t, masks_t)

            loss, v_loss, p_loss = self.criterion(logits, val_pred, pis_t, vals_t)
            loss.backward()

            # Gradient clipping
            if self.train_config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.train_config.gradient_clip_norm
                )

            self.optimizer.step()
            self.total_steps += 1

            total_loss_accum += loss.item()
            val_loss_accum += v_loss.item()
            pol_loss_accum += p_loss.item()
            actual_batches += 1

        if actual_batches == 0:
            return {"loss_total": 0.0, "loss_value": 0.0, "loss_policy": 0.0}

        return {
            "loss_total": total_loss_accum / actual_batches,
            "loss_value": val_loss_accum / actual_batches,
            "loss_policy": pol_loss_accum / actual_batches,
            "total_optimization_steps": self.total_steps,
        }
