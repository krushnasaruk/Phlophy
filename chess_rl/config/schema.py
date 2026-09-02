"""
Configuration schema definitions using typed dataclasses.
Provides strict validation and hierarchical experiment specifications.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class ExperimentMetaConfig:
    name: str = "baseline_experiment"
    description: str = "Chess RL experiment"
    paradigm: str = "direct"  # "direct" | "curriculum"
    seed: int = 42
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    output_dir: str = "experiments/results"


@dataclass
class RulesConfig:
    allow_castling: bool = True
    allow_en_passant: bool = True
    allow_promotion: bool = True
    fifty_move_rule: bool = True
    threefold_repetition: bool = True


@dataclass
class RewardSchemeConfig:
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    step_penalty: float = 0.0


@dataclass
class EnvironmentConfig:
    name: str = "full_chess_8x8"
    board_type: str = "standard_8x8"
    board_size: List[int] = field(default_factory=lambda: [8, 8])
    num_input_channels: int = 19
    action_space_size: int = 4096
    max_moves: int = 200
    pieces: Optional[Dict[str, List[str]]] = None
    rules: RulesConfig = field(default_factory=RulesConfig)
    reward_scheme: RewardSchemeConfig = field(default_factory=RewardSchemeConfig)


@dataclass
class ModelConfig:
    name: str = "resnet_small"
    architecture_type: str = "residual_cnn"
    num_residual_blocks: int = 4
    num_channels: int = 64
    kernel_size: int = 3
    padding: int = 1
    policy_head_channels: int = 2
    value_head_channels: int = 1
    value_hidden_dim: int = 64
    use_batch_norm: bool = True
    dropout_rate: float = 0.0
    activation: str = "relu"


@dataclass
class MCTSConfig:
    num_simulations: int = 100
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temp_threshold_move: int = 15


@dataclass
class ReplayBufferConfig:
    capacity: int = 20000
    min_samples_to_train: int = 500
    batch_size: int = 128
    prioritized: bool = False


@dataclass
class EvaluationConfig:
    eval_interval_iters: int = 5
    eval_games: int = 20
    baseline_opponents: List[str] = field(default_factory=lambda: ["random", "pure_mcts"])


@dataclass
class CheckpointingConfig:
    save_interval_iters: int = 5
    keep_best_checkpoint: bool = True


@dataclass
class TrainingConfig:
    num_iterations: int = 100
    games_per_iteration: int = 50
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    training_epochs_per_iter: int = 5
    gradient_clip_norm: float = 1.0
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)


@dataclass
class LRSchedulerConfig:
    type: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-5


@dataclass
class LossWeightsConfig:
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    regularization_weight: float = 1e-4


@dataclass
class OptimizerConfig:
    optimizer_type: str = "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)


@dataclass
class CurriculumStageConfig:
    stage_id: int = 0
    name: str = "stage_0"
    env_config: str = "configs/environment/full_chess_8x8.yaml"
    target_win_rate: Optional[float] = None
    min_games: int = 100
    max_games: int = 1000


@dataclass
class CurriculumConfig:
    enabled: bool = False
    transition_strategy: str = "performance_triggered"  # "fixed_steps" | "performance_triggered"
    stages: List[CurriculumStageConfig] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    experiment: ExperimentMetaConfig = field(default_factory=ExperimentMetaConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    raw_config: Dict[str, Any] = field(default_factory=dict)
