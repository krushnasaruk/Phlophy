# Scientific Experiment Plan: Curriculum Self-Play Chess RL

**Author:** Autonomous Chess Research Project  
**Target:** Bachelor of Engineering Thesis / Research Project  
**Status:** Planned Experimental Protocol (Phase 0 Specification)  

---

## 1. Overview and Core Research Objective

This document formalizes the experimental protocol designed to empirically evaluate the hypothesis that curriculum-based self-play reinforcement learning improves sample efficiency and stability compared to direct self-play.

> [!IMPORTANT]
> **No Fabricated Results:** This document defines the *experimental plan, execution protocols, and metrics definition*. Empirical numbers will be populated solely through real experimental execution in later phases.

---

## 2. Core Experimental Comparison

```
+-----------------------------------------------------------------------------------+
|                            EXPERIMENTAL PROTOCOL MATRIX                           |
+-----------------------------------+-----------------------------------------------+
| Experiment A (Direct Baseline)    | Experiment B (Curriculum Training)            |
+-----------------------------------+-----------------------------------------------+
| [Random Initialization]           | [Random Initialization]                       |
|         |                         |         |                                     |
|         v                         |         v                                     |
| [Full 8x8 Chess Environment]      | Stage 1: Basic Endgame (e.g., KQK / KRK)      |
|         |                         |         |                                     |
|         v                         |         v                                     |
| [Iterative Self-Play + MCTS]      | Stage 2: Mini-Chess (e.g., 5x5 Gardner/Silver)|
|         |                         |         |                                     |
|         v                         |         v                                     |
| [Fixed Compute Budget: N games]   | Stage 3: Full 8x8 Chess Environment           |
|                                   |         |                                     |
|                                   |         v                                     |
|                                   | [Equal Total Compute Budget: N games]         |
+-----------------------------------+-----------------------------------------------+
```

### Experiment A: Direct Training Baseline
- **Description:** The policy-value neural network is initialized with random weights and trained directly on the standard 8x8 chess environment through self-play and MCTS.
- **Environment:** Standard 8x8 Chess (64 squares, 32 pieces, full FIDE legal move logic).
- **Training Horizon:** Total budget of $T_{total} = 1,000$ self-play iterations ($K$ games per iteration).
- **Control Strategy:** Serves as the uncurriculum benchmark representing traditional AlphaZero-style learning from scratch.

### Experiment B: Curriculum-Based Training
- **Description:** The agent begins training on simplified environments with dense terminal rewards, progressively transitioning to more complex spaces as competence is verified.
- **Stage Progression:**
  - **Stage 1 (Endgame Mastery):** King + Queen vs. King (KQK) and King + Rook vs. King (KRK) on an 8x8 board with random initial placements. Target: Learn basic mate mechanics and short horizon rewards ($< 20$ moves).
  - **Stage 2 (Tactical Micro-Board):** Mini-Chess (5x5 or 6x6 reduced grid, subset of pieces: King, Queen, Rook, Bishop, Pawns). Target: Learn tactical interactions, piece trades, and board vision with reduced state space ($10^{18}$ vs $10^{44}$).
  - **Stage 3 (Full 8x8 Chess):** Standard 8x8 Chess initialized with the network weights transferred from Stage 2.
- **Curriculum Budget Allocation:**
  - Stage 1: $15\%$ of total compute budget.
  - Stage 2: $25\%$ of total compute budget.
  - Stage 3: $60\%$ of total compute budget.
  - *Total budget across all stages strictly equals Experiment A budget.*

---

## 3. Experimental Parameters & Budgets

| Parameter | Baseline A (Direct) | Curriculum B (Staged) | Ablation Variations |
| :--- | :--- | :--- | :--- |
| **Total Self-Play Games** | 50,000 games | 50,000 games (distributed) | 25,000 / 100,000 |
| **MCTS Simulations per Move** | 100 simulations | 100 simulations | 25, 50, 200 |
| **Dirichlet Noise ($\alpha$)** | 0.3 | 0.3 | 0.15, 0.45 |
| **Exploration Constant ($c_{puct}$)** | 1.25 | 1.25 | 1.0, 2.5 |
| **Replay Buffer Size** | 20,000 states | 20,000 states | 5,000, 50,000 |
| **Batch Size** | 128 | 128 | 64, 256 |
| **Optimizer** | AdamW ($\text{lr}=10^{-3}$) | AdamW ($\text{lr}=10^{-3}$) | SGD with momentum |
| **Network Architecture** | 4-Block Residual CNN | 4-Block Residual CNN | 2-Block / 8-Block |
| **Random Seeds** | 5 seeds (e.g., 42, 101, 2024, 7, 999) | 5 seeds (identical) | Identical seed pairs |

---

## 4. Evaluation and Tournament Protocol

To measure true learning progress without relying on external engines:

1. **Periodic Checkpointing:** Model weights are saved every 25 iterations.
2. **Internal Benchmark Arena:**
   - Every checkpoint plays a 100-game match against:
     1. Random-move agent (sanity baseline).
     2. Pure MCTS (rollouts without neural guidance).
     3. Historical checkpoint from iteration $i - 25$.
     4. Direct cross-evaluation: Checkpoint at budget $t$ in Experiment B vs. Checkpoint at budget $t$ in Experiment A.
3. **Elo Rating Estimation:**
   - Elo ratings are updated using Bayesian Elo / Glicko-2 estimation based on tournament match outcomes.
4. **Endgame & Tactical Accuracy Test Suite:**
   - 100 fixed tactical positions evaluated under fixed search budgets to measure:
     - First-choice move accuracy (top-1 policy agreement with winning move).
     - Success rate in finding forced checkmates within depth $D$.

---

## 5. Planned Ablation Studies

Following the core comparison (Exp A vs Exp B), the research will conduct systematic ablations to isolate contributing factors:

### Ablation 1: Curriculum Schedule Strategy
- **A1.1:** Fixed-Iteration Schedule (Fixed game counts per stage).
- **A1.2:** Performance-Triggered Schedule (Stage advance triggered when win-rate against reference pool $\ge 80\%$).
- **A1.3:** Reverse Curriculum (Starting from complex to simple - negative control).

### Ablation 2: Neural Network Capacity
- **A2.1:** Compact ResNet (2 residual blocks, 64 channels) — optimized for embedded microcontrollers / edge robotics.
- **A2.2:** Standard ResNet (4 residual blocks, 128 channels).
- **A2.3:** Deep ResNet (8 residual blocks, 256 channels).

### Ablation 3: MCTS Simulation Scaling
- **A3.1:** Fast Search ($N_{sim} = 25$).
- **A3.2:** Balanced Search ($N_{sim} = 100$).
- **A3.3:** Deep Search ($N_{sim} = 200$).
- **A3.4:** Adaptive Search (High simulations in high-entropy states, low simulations in forced lines).

---

## 6. Telemetry, Logging, and Reproducibility

Every experimental run must automatically produce the following artifact hierarchy:

```
experiments/results/
└── exp_<experiment_id>_<timestamp>_<seed_hash>/
    ├── config.yaml          # Full serialized configuration with seed
    ├── system_info.json     # Hardware specs (CPU, GPU, RAM, OS, PyTorch build)
    ├── training_metrics.csv # Step-by-step telemetry (losses, game lengths, win rates)
    ├── eval_results.json    # Tournament results and tactical benchmark accuracies
    ├── checkpoints/         # Model weights (.pt files) at scheduled intervals
    │   ├── model_iter_0025.pt
    │   ├── model_iter_0050.pt
    │   └── model_best.pt
    └── plots/               # Automatically rendered loss, Elo, and sample efficiency curves
```

### Reproducibility Verification Checklist
- [x] Deterministic seed initialization across `random`, `numpy`, and `torch`.
- [x] Version pinning in `requirements.txt` and `pyproject.toml`.
- [x] Automated system telemetry snapshot upon experiment launch.
- [x] Immutable experiment directory creation with timestamp and unique hash to prevent overwrite.
