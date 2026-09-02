"""
Configuration module for chess_rl.
"""

from chess_rl.config.schema import (
    ExperimentConfig,
    ExperimentMetaConfig,
    EnvironmentConfig,
    RulesConfig,
    RewardSchemeConfig,
    ModelConfig,
    TrainingConfig,
    MCTSConfig,
    ReplayBufferConfig,
    EvaluationConfig,
    CheckpointingConfig,
    OptimizerConfig,
    LRSchedulerConfig,
    LossWeightsConfig,
    CurriculumConfig,
    CurriculumStageConfig,
)
from chess_rl.config.loader import load_experiment_config

__all__ = [
    "ExperimentConfig",
    "ExperimentMetaConfig",
    "EnvironmentConfig",
    "RulesConfig",
    "RewardSchemeConfig",
    "ModelConfig",
    "TrainingConfig",
    "MCTSConfig",
    "ReplayBufferConfig",
    "EvaluationConfig",
    "CheckpointingConfig",
    "OptimizerConfig",
    "LRSchedulerConfig",
    "LossWeightsConfig",
    "CurriculumConfig",
    "CurriculumStageConfig",
    "load_experiment_config",
]
