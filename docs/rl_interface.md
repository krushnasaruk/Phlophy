# Reinforcement Learning Interface Contract: Observation, Action Space, and Representation

**Author:** Autonomous Chess Research Project  
**Target:** Bachelor of Engineering Thesis / Research Project  
**Status:** Formal Specification (Phase 2)  

---

## 1. Overview and Architectural Boundary

In this tabula-rasa self-play reinforcement learning framework, the interface between the discrete chess rules engine (`GameState`) and the deep neural network ($f_\theta(s) = (\mathbf{p}, v)$) is strictly defined by three components:
1. **Observation Tensor ($s$):** A continuous spatial representation of the board state.
2. **Fixed Action Space ($\mathcal{A}$):** A fixed-dimension discrete action indexing covering all legal chess transitions.
3. **Legal Action Mask ($\mathbf{m}$):** A boolean vector identifying admissible moves from the current position.

```
+-----------------------------------------------------------------------------------+
|                           RL INTERFACE CONTRACT DIAGRAM                           |
+-----------------------------------------------------------------------------------+

   +--------------------------+
   |   GameState (64-Board)   |
   +-------------+------------+
                 |
                 +--------------------------------+
                 |                                |
                 v (Tensor Encoding)              v (Legal Move Generation)
   +--------------------------+     +--------------------------+
   |    Observation Tensor    |     |    Legal Action Mask     |
   |      Shape: (19, 8, 8)   |     |      Shape: (4240,)      |
   |      Dtype: float32      |     |      Dtype: bool         |
   +-------------+------------+     +-------------+------------+
                 |                                |
                 v (Forward Pass)                 |
   +--------------------------+                   |
   | Dual-Head Neural Network |                   |
   |   Raw Logits: (4240,)    |                   |
   |   Value Pred: (1,)       |                   |
   +-------------+------------+                   |
                 |                                |
                 +----------------+---------------+
                                  |
                                  v (Numerically Stable Masking)
                    +--------------------------+
                    |  Masked Action Probs pi  |
                    |    pi_a = Softmax(Logit) |
                    +-------------+------------+
                                  |
                                  v (Action Decoding)
                    +--------------------------+
                    |  Executable Chess Move   |
                    |     Move(from, to)       |
                    +--------------------------+
```

---

## 2. Observation Tensor Specification

- **Shape:** `(19, 8, 8)` (or batched `(B, 19, 8, 8)`)
- **Data Type:** `numpy.float32` / `torch.float32`
- **Coordinate Alignment:** Spatial grid $(H, W)$ maps directly to $(\text{rank}, \text{file})$ with rank $0 \dots 7$ (Ranks 1 to 8) and file $0 \dots 7$ (Files 'a' to 'h').

### Feature Plane Breakdown

| Plane Index | Semantics | Value Range | Spatial Layout |
| :--- | :--- | :--- | :--- |
| **0** | White Pawns | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if White Pawn present |
| **1** | White Knights | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if White Knight present |
| **2** | White Bishops | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if White Bishop present |
| **3** | White Rooks | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if White Rook present |
| **4** | White Queens | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if White Queen present |
| **5** | White King | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if White King present |
| **6** | Black Pawns | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if Black Pawn present |
| **7** | Black Knights | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if Black Knight present |
| **8** | Black Bishops | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if Black Bishop present |
| **9** | Black Rooks | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if Black Rook present |
| **10** | Black Queens | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if Black Queen present |
| **11** | Black King | $\{0.0, 1.0\}$ | $1.0$ at square $(r, f)$ if Black King present |
| **12** | White Kingside Castling | $\{0.0, 1.0\}$ | Uniform plane: all $1.0$ if White has O-O rights, $0.0$ otherwise |
| **13** | White Queenside Castling | $\{0.0, 1.0\}$ | Uniform plane: all $1.0$ if White has O-O-O rights, $0.0$ otherwise |
| **14** | Black Kingside Castling | $\{0.0, 1.0\}$ | Uniform plane: all $1.0$ if Black has O-O rights, $0.0$ otherwise |
| **15** | Black Queenside Castling | $\{0.0, 1.0\}$ | Uniform plane: all $1.0$ if Black has O-O-O rights, $0.0$ otherwise |
| **16** | Active Side to Move | $\{-1.0, +1.0\}$ | Uniform plane: all $+1.0$ if White's turn, all $-1.0$ if Black's turn |
| **17** | En Passant Target Square | $\{0.0, 1.0\}$ | $1.0$ at coordinate $(r, f)$ if en passant target active, $0.0$ otherwise |
| **18** | Halfmove Clock (50-Move) | $[0.0, 1.0]$ | Uniform plane: $\min(\text{halfmove\_clock} / 100.0, 1.0)$ |

---

## 3. Discrete Action Space Specification

- **Total Action Space Size ($|\mathcal{A}|$):** `4,240`
- **Index Range:** `0 <= action_id < 4240`
- **Mathematical Bijection:** Every distinct legal move $(from, to, promo)$ maps to exactly one discrete integer index.

### Action Index Mapping Structure

```
[ 0 ------------------------------------ 4095 ] [ 4096 ---------------------------- 4239 ]
  Normal Moves, Captures, Castling,               Underpromotions (Knight, Bishop, Rook)
  En Passant, and Default Queen Promotions        48 Knight + 48 Bishop + 48 Rook Slots
```

1. **Standard Transitions ($0 \dots 4095$):**
   $$\text{action\_id} = \text{from\_square} \times 64 + \text{to\_square}$$
   - Covers all normal piece steps, leaps, sliding moves, and captures.
   - Castling moves (White O-O: $e1g1 = 4 \times 64 + 6 = 262$, White O-O-O: $e1c1 = 4 \times 64 + 2 = 258$).
   - En passant captures (e.g., $e5d6 = 36 \times 64 + 43 = 2347$).
   - Default Queen promotions ($e7e8q = 52 \times 64 + 60 = 3388$).

2. **Underpromotion Transitions ($4096 \dots 4239$):**
   - Pawns reaching the promotion rank (Rank 8 for White, Rank 1 for Black) can promote to Knight, Bishop, or Rook.
   - For each pawn file ($0 \dots 7$) and allowable destination file ($\Delta f \in \{-1, 0, +1\}$), dedicated unique slots exist:
     - Knight Underpromotions: Indices $4096 \dots 4143$ ($48$ slots)
     - Bishop Underpromotions: Indices $4144 \dots 4191$ ($48$ slots)
     - Rook Underpromotions: Indices $4192 \dots 4239$ ($48$ slots)

---

## 4. Legal Action Masking Specification

- **Shape:** `(4240,)` (or batched `(B, 4240)`)
- **Data Type:** `numpy.bool_` / `torch.bool`
- **Mask Invariants:**
  1. $\mathbf{m}[a] = \text{True} \iff a \text{ corresponds to a strictly legal move in current state}$.
  2. $\sum_{a=0}^{4239} \mathbf{m}[a] = |\text{legal\_moves}|$.
  3. For terminal states (Checkmate, Stalemate), $\sum \mathbf{m}[a] = 0$.

### Numerical Stability Masking Formula
Given raw network policy logits $\mathbf{z} \in \mathbb{R}^{4240}$ and binary mask $\mathbf{m} \in \{0, 1\}^{4240}$:
$$\tilde{z}_a = \begin{cases} z_a & \text{if } m_a = 1 \\ -1\times 10^9 & \text{if } m_a = 0 \end{cases}$$
$$\pi(a) = \text{Softmax}(\tilde{\mathbf{z}})_a = \frac{\exp(\tilde{z}_a)}{\sum_{b} \exp(\tilde{z}_b)}$$

---

## 5. Current Environment Reward Semantics

In Phase 1 and Phase 2, reward signals are strictly terminal to preserve tabula-rasa learning integrity:

$$R(\text{terminal}, \text{player}) = \begin{cases} 
+1.0 & \text{if } \text{player wins by Checkmate} \\
-1.0 & \text{if } \text{player loses by Checkmate} \\
 0.0 & \text{if game ends in Draw (Stalemate, 50-move, Repetition, Insufficient Material)}
\end{cases}$$

For intermediate (non-terminal) steps, $R(s, a) = 0.0$.
