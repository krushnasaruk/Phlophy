# System Architecture: End-to-End Self-Play RL & Autonomous Chessboard Pipeline

**Author:** Autonomous Chess Research Project  
**Target:** Bachelor of Engineering Thesis / Research Project  
**Status:** System Architecture Specification (Phase 0)  

---

## 1. High-Level Software Architecture

The software architecture is engineered to be strictly modular, decoupled, and scientific. It separates environment mechanics, tensor representations, neural networks, tree search, self-play data generation, optimization, evaluation, and future physical hardware interfaces.

```
+---------------------------------------------------------------------------------------------------------+
|                                    CHESS RL SOFTWARE PIPELINE                                           |
+---------------------------------------------------------------------------------------------------------+

  [ Chess Environment ] (Full 8x8 / Mini-Chess / Endgame)
           |
           v (State Observation)
  [ Board Representation ] (Tensor Encoding: C x H x W + Legal Action Mask)
           |
           v (Observation Tensor)
  [ Neural Network (Policy-Value) ] (Shared Residual Backbone -> Policy Head & Value Head)
           |
           v (Priors p, Value v)
  [ Monte Carlo Tree Search (MCTS) ] (PUCT Selection -> Expansion -> Backup -> Visit Distribution)
           |
           v (Improved Policy pi)
  [ Move Selection Engine ] (Temperature Sampling / Argmax)
           |
           +-------------------------------------------------------------+
           | (Self-Play Loop)                                            | (Inference / Tournament / Robot)
           v                                                             v
  [ Self-Play Episode Runner ]                                 [ AI Move Decision (UCI/Coord) ]
           |                                                             |
           v (Episode History: s, pi, z)                                 v
  [ Experience Replay Buffer ]                                 [ Hardware Controller Interface ]
           |                                                             |
           v (Batches of Transitions)                                    v
  [ Optimization & Training Loop ] (AdamW, Loss = MSE + CE + L2)  [ Physical Chessboard Actuator ]
           |
           v (Updated Model Checkpoint)
  [ Evaluation Arena & Model Registry ] (Tournament matches, SPRT, Elo tracking, Checkpoint Selection)
           |
           v
  [ Next-Generation Agent ]
```

---

## 2. Core Subsystems

### 2.1 Environment Subsystem (`chess_env/`)
- **`BaseChessEnvironment`:** Abstract interface defining `reset()`, `step(action)`, `legal_actions()`, `is_terminal()`, `get_reward()`, and `render()`.
- **Environment Variants:**
  - `StandardChessEnv`: Standard 8x8 FIDE rules.
  - `MiniChessEnv`: 5x5 or 6x6 spatial subset with reduced piece sets.
  - `EndgameEnv`: Algorithmic subset initializations (e.g., KQK, KRK).
- **Decoupled Mechanics:** All environment logic is pure Python/NumPy, with no external engine reliance.

### 2.2 Board Representation Layer (`chess_env/representation/`)
- Encodes discrete board states into standardized $C \times H \times W$ float32 tensors suitable for convolutional and attention backbones:
  - **Planes 0–5:** White piece positions (P, N, B, R, Q, K).
  - **Planes 6–11:** Black piece positions (P, N, B, R, Q, K).
  - **Planes 12–15:** Castling rights (White kingside/queenside, Black kingside/queenside).
  - **Plane 16:** Active player turn indicator ($+1$ for White, $-1$ for Black).
  - **Plane 17:** Move count / 50-move rule counter normalized to $[0, 1]$.
  - **Plane 18:** Repetition counter plane.
- **Action Space Encoding:** Flattened discrete action vector of dimension $|\mathcal{A}|$ (e.g., $64 \times 64 = 4096$ coordinate-based source-to-target move index or 4672 AlphaZero-style plane representation).

### 2.3 Agent & Neural Network Layer (`agents/`)
- **`BaseAgent`:** Abstract interface for selecting moves given an environment state.
- **`BasePolicyValueNet` (PyTorch):**
  - **Backbone:** Convolutional input layer followed by $N$ Residual Blocks with Batch Normalization and ReLU activations.
  - **Policy Head:** $1 \times 1$ Conv $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Linear $\rightarrow$ LogSoftmax over valid actions.
  - **Value Head:** $1 \times 1$ Conv $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Linear ($128$) $\rightarrow$ ReLU $\rightarrow$ Linear ($1$) $\rightarrow$ Tanh outputting $v \in [-1, +1]$.

### 2.4 Search Subsystem (`search/mcts/`)
- Implements parallelizable, high-efficiency Monte Carlo Tree Search.
- Tree nodes store visit counts $N(s, a)$, cumulative action-value $W(s, a)$, mean action-value $Q(s, a)$, prior probability $P(s, a)$, and children links.
- Uses vectorized legal action masking to eliminate illegal subtree exploration.

### 2.5 Training & Curriculum Subsystem (`training/`)
- **`SelfPlayWorker`:** Executes parallel episodes using current network checkpoints and MCTS, accumulating trajectory tuples $(s_t, \boldsymbol{\pi}_t, z)$.
- **`ReplayBuffer`:** Uniform and prioritized circular memory buffers for off-policy batch sampling.
- **`CurriculumManager`:** Manages environmental progression based on performance metrics (win rates, episode lengths, tactical pass rates).
- **`Trainer`:** Executes mini-batch gradient descent, tracks learning rate schedules, computes gradient norms, and triggers checkpointing.

### 2.6 Evaluation Subsystem (`evaluation/`)
- **`Arena`:** Executes head-to-head tournaments between competing agent generations or benchmark bots.
- **`MetricsEvaluator`:** Computes Elo, Glicko-2 ratings, win/loss/draw rates, and statistical confidence intervals ($p$-values).
- **`BenchmarkSuite`:** Evaluates agents on fixed endgame and tactical puzzle banks.

---

## 3. Physical Hardware Interface Architecture (Future Deployment)

To prepare for future deployment onto an autonomous robotic chessboard, the software architecture incorporates an abstracted hardware interface layer. The software intelligence is completely decoupled from low-level mechanical kinematics.

```
+-------------------------------------------------------------------------------------+
|                             PHYSICAL DEPLOYMENT STACK                               |
+-------------------------------------------------------------------------------------+

  [ Autonomous Chess RL Agent ]
                |
                v (Selects Action: e.g., Move(from_sq="e2", to_sq="e4", promo=None))
  [ Move Command Protocol (JSON / UCI Coordinate Formatter) ]
                |
                v (Serialized Move Packet)
  [ Hardware Controller Interface (BaseHardwareController) ]
                |
                +------------------------------------+--------------------------------+
                |                                    |                                |
                v                                    v                                v
    [ CoreXY / Gantry Kinematics ]        [ SCARA / 6-DOF Arm Kinematics ]  [ Magnetic Sensor Grid ]
    (Converts square to (X, Y) mm)        (Computes Joint Angles theta)      (Hall effect / Reed switch)
                |                                    |                                |
                v                                    v                                v
    [ Stepper Motor Actuators ]           [ Servo / Stepper Drivers ]       [ Board State Detector ]
    (Electromagnet Grab / Release)        (Gripper Pick / Place)             (Detects Human Player Move)
                |                                    |                                |
                +------------------------------------+--------------------------------+
                                                     |
                                                     v
                                       [ Physical Chessboard Surface ]
```

### Hardware Abstraction Layer Specification
- **`BaseHardwareController` (Abstract Base Class):**
  - `connect(port: str, baudrate: int) -> bool`
  - `send_move(move_from: str, move_to: str, capture: bool, promotion: Optional[str]) -> bool`
  - `query_board_state() -> Optional[str]` (FEN string or square occupancy matrix)
  - `emergency_stop() -> None`
- **Decoupling Guarantee:** The RL agent emits standard abstract actions (e.g., `e2e4`); the hardware translation layer maps this move into Cartesian coordinate paths, actuator acceleration profiles, and electromagnet activation sequences without altering the agent codebase.
