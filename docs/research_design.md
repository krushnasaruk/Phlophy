# Research Design: Compute-Efficient Curriculum-Based Self-Play Reinforcement Learning for Autonomous Chess

**Author:** Autonomous Chess Research Project  
**Target:** Bachelor of Engineering Thesis / Research Project  
**Status:** Foundational Research Specification (Phase 0)  

---

## 1. Executive Summary

This document establishes the scientific research foundation for developing an autonomous chess agent that learns entirely **from scratch** using reinforcement learning (RL), self-play, and Monte Carlo Tree Search (MCTS), without relying on human knowledge, pre-existing game databases, external engines (e.g., Stockfish), or pre-trained models.

The central research inquiry investigates whether a **progressive curriculum of simplified chess environments and tactical stages** can significantly reduce the computational budget and sample complexity required to reach stable chess competence compared to standard direct self-play on the full 64-square board.

---

## 2. Research Problem

Standard self-play reinforcement learning algorithms (such as AlphaZero) demonstrate that tabular and neural agents can achieve superhuman mastery in board games from zero initial knowledge. However, applying these algorithms directly to standard 8x8 chess exhibits severe engineering and computational challenges:

1. **Massive Sample Inefficiency in Cold-Start Phase:** At initial random initialization, games are characterized by aimless wandering and extreme game lengths (often reaching 200+ moves before reaching random draws or blundered checkmates). The reward signal (+1 for win, -1 for loss, 0 for draw) is extremely sparse and delayed across hundreds of plies.
2. **Extreme Compute Requirements:** AlphaZero required thousands of TPUs and tens of millions of self-play games. In academic, collegiate, and resource-constrained environments, reproducing such scale is infeasible.
3. **Training Instability and Value Collapse:** Sparse terminal rewards under high branching factors (~35 legal moves per position) frequently cause early value network divergence, policy gradient variance explosions, or policy cycling.

The fundamental research problem is:  
> *How can an agent learn optimal chess strategies from scratch within a constrained computational budget while mitigating sample inefficiency and training instability?*

---

## 3. Research Gap

While AlphaZero established the paradigm of tabula-rasa self-play, subsequent research has predominantly focused on:
- Scaling up model parameter counts (e.g., Leela Chess Zero).
- Pure search optimizations without altering the learning trajectory.
- Generalized game playing across multiple games rather than sample-efficient curriculum learning for complex tactical games.

Existing literature lacks a systematic, rigorous comparative study investigating:
1. **Curriculum Design for From-Scratch Self-Play:** How structured micro-board variants (e.g., 5x5, Gardner Chess), endgame subsets (e.g., King+Queen vs. King, King+Rook vs. King), and progressive tactical stages impact policy convergence and value calibration.
2. **Transfer Learning Dynamics Across Dimensionalities:** How representations learned on smaller spatial grids or reduced piece counts transfer to the full 8x8 chess state-action space without catastrophic forgetting or negative transfer.
3. **Hardware Deployment Feasibility:** Designing lightweight, compute-efficient architectures capable of running in real-time inference loops on resource-limited embedded hardware suitable for physical autonomous chessboard robotics.

Simply re-implementing AlphaZero on an 8x8 board with small compute results in negligible learning due to reward sparsity. This research directly bridges that gap by innovating at the curriculum and compute-efficiency layer.

---

## 4. Research Hypotheses

### Primary Research Hypothesis ($H_1$)
> **$H_1$:** A structured, progressive curriculum consisting of simplified micro-board chess environments and endgame sub-problems significantly improves the sample efficiency (measured by games and gradient steps required to achieve fixed tactical/strategic milestones) and training stability (measured by policy/value loss variance and Elo progression monotonicity) of a self-play reinforcement learning agent trained from random initialization, compared to an identical agent trained directly on standard 8x8 chess under equal computational budgets.

### Null Hypothesis ($H_0$)
> **$H_0$:** A progressive curriculum provides no statistically significant improvement, or actively impairs the sample efficiency, training stability, or final playing strength of a self-play reinforcement learning agent compared to direct 8x8 self-play under equal computational budgets.

---

## 5. Experimental Variables

To maintain scientific validity, all experiments will adhere to rigorous variable isolation:

| Variable Category | Variable Name | Definition & Operationalization |
| :--- | :--- | :--- |
| **Independent Variables** | **Training Paradigm** | `Direct Baseline` (8x8 full chess from step 0) vs. `Progressive Curriculum` (Staged transition: Endgame $\rightarrow$ Mini-Chess $\rightarrow$ Full 8x8). |
| | **Curriculum Staging Policy** | Fixed-step schedule vs. Performance-triggered adaptive schedule (win-rate / tactical solve rate thresholds). |
| | **MCTS Simulation Budget ($N_{sim}$)** | Number of MCTS rollouts per move decision during self-play (e.g., $N \in \{50, 100, 200\}$). |
| | **Network Capacity** | Residual block depth and channel width (e.g., 4-block vs. 8-block ResNet). |
| **Dependent Variables** | **Sample Efficiency** | Total environment interactions and self-play games required to reach benchmark Elo and tactical puzzle solve thresholds. |
| | **Playing Strength (Elo)** | Relative Elo rating measured via round-robin / Swiss tournaments and Sequential Probability Ratio Tests (SPRT). |
| | **Training Stability** | Variance of moving-average value loss ($\mathcal{L}_{MSE}$) and policy cross-entropy loss ($\mathcal{L}_{CE}$) across iterations. |
| | **Compute Resource Usage** | Wall-clock time (seconds), total floating-point operations (FLOPs), peak GPU/CPU memory (MB/GB). |
| | **Tactical Accuracy** | Percentage of forced mate-in-1, mate-in-2, and decisive material captures correctly solved in standard test suites. |
| **Controlled Variables** | **Random Seed Suites** | Identical sets of pseudo-random seeds ($n \ge 5$) applied across weight initialization, MCTS noise, and environment initial states. |
| | **Loss Formulations** | Exact identical loss function $\mathcal{L} = (z - v)^2 - \boldsymbol{\pi}^\top \log \mathbf{p} + c\|\theta\|^2$. |
| | **Optimizer & Learning Rate** | AdamW with identical base learning rates, warmup schedules, and weight decay constants. |
| | **Replay Buffer Mechanics** | Uniform/prioritized replay buffer capacity, mini-batch size, and sampling ratios. |
| | **Search Parameters** | $c_{puct}$ exploration constant, Dirichlet noise parameter $\alpha$, and temperature schedule $\tau$. |
| | **Hardware Platform** | Execution on identical compute infrastructure to isolate hardware-dependent timing variance. |

---

## 6. Evaluation Framework & Statistical Rigor

1. **Replication:** Every experimental condition will be executed across a minimum of $N = 5$ independent random seeds.
2. **Statistical Significance Testing:**
   - **Welch's Two-Sample t-Test / Mann-Whitney U Test:** Used to compare sample efficiency metrics and final Elo distributions between Baseline and Curriculum conditions at a significance level of $\alpha = 0.05$.
   - **Confidence Intervals:** 95% bootstrap confidence intervals reported for all win-rates and Elo ratings.
3. **Benchmark Suites:**
   - **Endgame Solving Test Suite:** Set of canonical algorithmic endgame positions (e.g., K+Q vs K, K+R vs K) evaluated for optimal step-to-mate.
   - **Tactical Test Suite:** Cold-evaluation test suite containing 100 tactical positions with unique optimal solutions.
   - **Head-to-Head Arena Tournaments:** Direct match play between checkpoint generations using opening book randomization (neutral symmetry) to eliminate first-mover advantage bias.

---

## 7. Ethical and Scientific Integrity

- **Zero Contamination:** No pretrained neural weights, chess engines, opening books, or human game records will enter the training loop.
- **Full Reproducibility:** Every experiment logs the complete configuration, system metadata, seed values, loss trajectories, and model checkpoints.
- **Reporting Standard:** All results—including negative results or failed curriculum transitions—will be faithfully reported without selective omission.
